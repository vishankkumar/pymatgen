import sys
import os
import numpy as np
from pymatgen.io.abinit.works import Work, RelaxWork
from pymatgen.core.periodic_table import Element
from numpy import random
from abipy import abilab
import logging

logger = logging.getLogger(__name__)


class MDWork(Work):
    """
    Work for combining structural relaxation with molecular dynamics. The first task relaxes the atomic position
    while keeping the unit cell parameters fixed. The second task uses the final
    structure to perform a structural relaxation in which both the atomic positions
    and the lattice parameters are optimized. The third task performs MD calculations
    """
    counter = 0
    _DEFAULT_PARAMS = dict(pseudos=None, ecut = 35, ngkpt = np.array((2,2,2)), occopt = 1, 
        tsmear = 0.01, nshiftk = 1, shiftk = np.array((0.0,0.0,0.0)), element_rem = "Li", target_count = 3, tolmxf = 1e-6, tolvar='toldff',tolval=1e-7, 
        optcell = 2,mdtime = 1, md_temp = 1000, md_ionmov = 12)

    
    @classmethod
    def from_structure(cls, structure=None,workdir=None, manager=None, params=None):
        """
        Args:
            ion_input: Input for the relaxation of the ions (cell is fixed)
            md_input: Input for the relaxation of the ions and the unit cell.
            workdir: Working directory.
            manager: :class:`TaskManager` object.
        """

        first = MDWork(workdir=workdir,manager=manager)
        first.params = cls._DEFAULT_PARAMS.copy()
        if params is not None:
            first.params.update(**params)
        
        
        #super(MDWork, self).__init__(workdir=workdir, manager=manager)
        ion_input,ioncell_input = first.make_ion_ioncell_input(structure)
        md_input = first.make_md_input(structure)

        deps = None
        first.ion_task = first.register_relax_task(ion_input)
        first.ioncell_task = first.register_relax_task(ioncell_input)
        first.md_task = first.register_relax_task(md_input)
        
        first.ioncell_task.add_deps ({first.ion_task: "@structure"})
        first.md_task.add_deps ({first.ioncell_task: "@structure"})
        
        first.transfer_done = False
        first.new_work_created = False
        return first
 


    @classmethod
    def count(cls):
        cls.counter+=1
        print ('Structure created {}'.format(cls.counter))
    
    
    def make_ion_ioncell_input(self,structure=None):
        nelec = structure.num_valence_electrons(pseudos=self.params['pseudos']) 
        print (nelec) 
        nbdbuf=np.ceil(0.05*nelec/2)
        print (nbdbuf) 
        temp = int(nelec/2+nbdbuf+8)
        global_vars = dict(
            ecut=self.params['ecut'],
            ngkpt=self.params['ngkpt'], 
            shiftk=self.params['shiftk'],
            nshiftk=self.params['nshiftk'],
    	    chkprim=0,
            chksymbreak=0,
            occopt=self.params['occopt'],
            tsmear=self.params['tsmear'],
            paral_kgb=1,
            prtwf=-1,
            prtden=0,
            nbdbuf=nbdbuf,
            nstep = 100,
            ionmov = 22,
            nband = temp -temp%4
 
        )
        global_vars[self.params['tolvar']]=self.params['tolval'],
    
        multi = abilab.MultiDataset(structure, pseudos=self.params['pseudos'], ndtset=2)
    
        # Global variables
        multi.set_vars(global_vars)
    
        # Dataset 1 (Atom Relaxation)
        multi[0].set_vars(
            optcell=0,
            tolmxf=self.params['tolmxf'],
            ntime=200
        )
    
        # Dataset 2 (Atom + Cell Relaxation)
        multi[1].set_vars(
            optcell=self.params['optcell'],
            ecutsm=0.5,
            dilatmx=1.1,
            tolmxf=self.params['tolmxf'],
            strfact=100,
            ntime=200
            )
    
        ion_inp, ioncell_inp = multi.split_datasets()
        return ion_inp, ioncell_inp
    
    def make_md_input(self,structure=None):
        """Creates the input for MD run from the structure"""
        inp = abilab.AbinitInput(structure=structure,
            pseudos=self.params['pseudos'])
        #nelec = structure.num_valence_electrons(pseudos=pseudos)#"../pseudos/14si_new_GGA_PBE.pspnc")
        nelec = structure.num_valence_electrons(pseudos=self.params['pseudos']) 
        nbdbuf=np.ceil(0.05*nelec/2)
        temp = int(nelec/2+nbdbuf+8)
        inp.set_vars(
            #nbdbuf=int(nbdbuf),
            ecut=self.params['ecut'],
    	    paral_kgb=1,
            nband = temp -temp%4,
            nstep=100,
            tolrff=self.params['tolrff'],
    	    chkprim=0,
            chksymbreak=0,
    	    nsym=1,
            kptopt=0, 
            nkpt=1, #self.params['ngkpt'],
    	    nshiftk=1, #self.params['nshiftk'],
    	    shiftk=[0.0, 0.0, 0.0], #self.params['shiftk'],
            prtden=0,
            prtwf=-1
            )
    	#Metallic system input variables
        inp.set_vars(
    	    ionmov = self.params['md_ionmov'],
    	    dtion = 200,
    	    mdtemp = [self.params['md_temp'],self.params['md_temp']],
    	    optcell = 0,
    	    occopt = self.params['occopt'],
    	    tsmear = [self.params['md_temp'],"k"],
    	    ntime = self.params['mdtime']
    	)
        return (inp)


    def remove_atom(self,structure,element):
        print (element)
        try:
            sites=[isite for isite, site in enumerate(structure) if Element(element) in site.species]
            rem_site=random.choice(sites)
            return structure.remove_sites([rem_site])
        except:
            print ("This is an error message")

    #@check_spectator
    def on_ok(self, sender):
        """
        This callback is called when one task reaches status S_OK.
        If sender == self.ion_task, we update the initial structure
        used by self.ioncell_task and we unlock it so that the job can be submitted.
        """
        logger.debug("in on_ok with sender %s" % sender)
        print ("in on_ok with sender of print %s" % sender)
        if sender == self.ion_task and not self.transfer_done:
            # Get the relaxed structure from ion_task
            ion_structure = self.ion_task.get_final_structure()
            
            # Transfer it to the ioncell task (we do it only once).
            self.ioncell_task._change_structure(ion_structure)
            self.transfer_done = True
    
            print ("Printing the sender name %s" % sender)
            # Unlock ioncell_task so that we can submit it.
            self.ioncell_task.unlock(source_node=self.ion_task)
    
        elif sender == self.ioncell_task: # and not self.transfer_md_done:
            #md_structure=self.ioncell_task.get_final_structure()
            #self.md_task._change_structure(md_structure)
            #self.transfer_md_done = True 
            
            print ("Printing the sender name %s" % sender)
            # Unlock md_task so that we can submit it.
            #self.md_task.unlock(source_node=self.ioncell_task)
            
        #elif sender == self.md_task:
        elif sender == self.md_task:
            if self.new_work_created:
                print ('New work has already been created, check your flow')
                sys.exit()
            new_structure = self.md_task.get_final_structure()
            num_sites = len([isite for isite, site in enumerate(new_structure) if Element(self.params['element_rem']) in site.species])
            print ("Printing the counter value %s" % MDWork.counter)
            print ("Printing the target count %s" % self.params['target_count'])
            print ('Number of atoms left to be removed: {}'.format(num_sites))
            if (sender == self.md_task):
            	print ("Sender matched with self.md_task")
            else :
            	print ("Sender did not match")
   
            if (MDWork.counter < self.params['target_count'] and num_sites > 0):
                print("Creating new work")
                #old_structure = self.md_task.get_final_structure()
                print ("Before removing")
                self.remove_atom(new_structure,self.params['element_rem'])
                print ("Removed")
                print (new_structure)
                print ("Printing the sender name %s" % sender)
                #print('Class Variables: mdtime {}, target_count {}, ecut {}, tolrff {}, tolmxf {}, ngkpt {}'.format(MDWork.mdtime,MDWork.target_count,MDWork.ecut,MDWork.tolrff,MDWork.tolmxf,MDWork.ngkpt))     
                logger.debug("Creating a new work mentioning in logger")
                print ('Youpiee check')
                work = MDWork.from_structure(structure=new_structure,params = self.params)
                try:
                #work = MDWork.from_strcture(structure=new_structure,params = None)
                    work[2].add_deps ({work[1]: "@structure"})
                    self.flow.register_work(work)
                    self.flow.allocate()	
                    self.flow.build_and_pickle_dump()
                    self.count() #MDWork.counter+=1
                    self.new_work_created = True
                    print ('New work created {}, counter value is: {}'.format(self.new_work_created, MDWork.counter))
                except:
                    print ("Error in creating a new work")
            else:
                print ('All work created, counter {} reached target count {}'.format(MDWork.counter,self.params['target_count']))
        return super(MDWork, self).on_ok(sender)

class my_MDWork(MDWork):

    _DEFAULT_PARAMS = dict(pseudos=None, ecut = 32, ngkpt = np.array((2,2,2)), occopt = 1, 
        #flow ={'task1':"ion",'task2':"ioncell",'task3':"md"},#'task4':"ion_relax",'task5':"ioncell_relax"},
        tsmear = 0.01, nshiftk = 1, shiftk = np.array((0.0,0.0,0.0)), element_rem = "Li", target_count = 3, tolmxf = 1e-6, tolvar='toldff',tolval=1e-7, 
        optcell = 2,mdtime = 1, md_temp = 1000, md_ionmov = 12)
    

    @classmethod
    def from_structure(cls, structure=None,workdir=None, manager=None, params=None):
        """
        Args:
            ion_input: Input for the relaxation of the ions (cell is fixed)
            md_input: Input for the relaxation of the ions and the unit cell.
            workdir: Working directory.
            manager: :class:`TaskManager` object.
        """
        
        first = my_MDWork(workdir=workdir,manager=manager)
        first.params = cls._DEFAULT_PARAMS.copy()
        if params is not None:
            first.params.update(**params)
         
        #super(my_MDWork, self).__init__(workdir=workdir, manager=manager)
        ion_input,ioncell_input = first.make_ion_ioncell_input(structure)
        md_input = first.make_md_input(structure)
        #tasks = {"ion":ion_input,'ioncell':ioncell_input,'md':md_input}

        #for i,item in enumerate(first.params['flow_seq']):    
            #setattr(first, "task"+str(i), first.register_relax_task(tasks[item]))
            #if i!=0:
            #    first.task
        
        #print (first.params['flow'])
        #for k,v in first.params['flow'].items():
        #    print (k,v)
        #    setattr(first, k, v)
        
        first.task1 = first.register_relax_task(ion_input)
        deps = None
        first.task2 = first.register_relax_task(ioncell_input)
        first.task3 = first.register_relax_task(md_input)
        first.task4 = first.register_relax_task(ion_input)
        first.task5 = first.register_relax_task(ioncell_input)
        
        first.new_work_created = False
        first.task2.add_deps({first.task1: "@structure"})
        first.task3.add_deps({first.task2: "@structure"})
        first.task4.add_deps({first.task3: "@structure"})
        first.task5.add_deps({first.task4: "@structure"})
        return first
 
    #@check_spectator
    def on_ok(self, sender):
        """
        This callback is called when one task reaches status S_OK.
        If sender == self.ion_task, we update the initial structure
        used by self.ioncell_task and we unlock it so that the job can be submitted.
        """
        logger.debug("in on_ok with sender %s" % sender)
        print ("in on_ok with sender of print %s" % sender)
        if sender == self.task1:
            print ("Printing the sender name %s" % sender)
    
        elif sender == self.task2:
            print ("Printing the sender name %s" % sender)
        
        elif sender == self.task3:
            print ("Printing the sender name %s" % sender)
        
        elif sender == self.task4:
            print ("Printing the sender name %s" % sender)
        
        elif sender == self.task5:
            print ("Sender matched with self.md_task")
            if self.new_work_created:
                print ('New work has already been created, check your flow')
                sys.exit()
            new_structure = self.task5.get_final_structure()
            num_sites = len([isite for isite, site in enumerate(new_structure) if Element(self.params['element_rem']) in site.species])
            print ("Printing the counter value %s" % my_MDWork.counter)
            print ("Printing the target count %s" % self.params['target_count'])
            print ('Number of atoms left to be removed: {}'.format(num_sites))
   
            if (my_MDWork.counter < self.params['target_count'] and num_sites > 0):
                print("Creating new work")
                #old_structure = self.md_task.get_final_structure()
                print ("Before removing")
                self.remove_atom(new_structure,self.params['element_rem'])
                print ("Removed")
                print (new_structure)
                work = my_MDWork.from_structure(structure=new_structure,params = self.params)
                try:
                    #work[2].add_deps ({work[1]: "@structure"})
                    self.flow.register_work(work)
                    self.flow.allocate()	
                    self.flow.build_and_pickle_dump()
                    self.count() #my_MDWork.counter+=1
                    self.new_work_created = True
                    print ('New work created {}, counter value is: {}'.format(self.new_work_created,my_MDWork.counter))
                except:
                    print ("Error in creating a new work")
            else:
                print ('All work created, counter {} reached target count {}'.format(my_MDWork.counter,self.params['target_count']))
        return super(MDWork,self).on_ok(sender)


