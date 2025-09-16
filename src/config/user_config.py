# This module stores user-adjustable configuration values.

from pathlib import Path

                                                                               
                                   
                                                                               

                                 
DATA_DIR = Path("./Data/raw")                                                                          
RESULTS_DIR = Path("./results/DRAFTS++")                                                          

                              
FRB_TARGETS = [
   "2017-04-03-12_56_05_230_0002_t2.3_t17.395",
   "2017-04-03-08_16_13_142_0006_t10.882_t25.829",
   "2017-04-03-08_55_22_153_0006_t23.444",
   "2017-04-03-08-16-13_142_0003_t39.977",
   "2017-04-03-12_56_05_230_0003_t36.548",
   "2017-04-03-13_38_31_242_0005_t44.169",
]


                                                                               
                                    
                                                                               

                                                
SLICE_DURATION_MS: float = 300.0

                                                                               
                               
                                                                               

                                                       
DOWN_FREQ_RATE: int = 1                                                                             
DOWN_TIME_RATE: int = 32                                                                        

                                                                               
                                  
                                                                               

                                           
DM_min: int = 0                                                   
DM_max: int = 4000                                                

                                                                               
                       
                                                                               

                                                       
DET_PROB: float = 0.3                                                                                 
CLASS_PROB: float = 0.5                                                                     

                                                
SNR_THRESH: float = 3.0                                                                     

                                                                               
                                       
                                                                               

                                      
USE_MULTI_BAND: bool = False                                                                               

                                                                               
                                                 
                                                                               

                                                               
                                                                           
                                                                                
POLARIZATION_MODE: str = "linear"

                                                          
POLARIZATION_INDEX: int = 0

                                                                               
                                  
                                                                               

                                 
DEBUG_FREQUENCY_ORDER: bool = True                                                                         
                                                                                                             

                                                                
FORCE_PLOTS: bool = True                                                                

                                                                               
                                         
                                                                               

                                                           
SAVE_ONLY_BURST: bool = True                                                                                         
