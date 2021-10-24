from numpy.core.numeric import Infinity
import pandas as pd
from runpy import run_path
import keras.backend as K

def Optimization(totalStrat, n):
    Rest = open("Calibracao.txt","w+")
    myfile = open('Instances.txt')
    myfile2 = open('Instances2.txt')
    next_line = myfile.readline()
    next_line2 = myfile2.readline()

    #while next_line != "" and next_line != "\n": 
    for i in range(0, n):
        arquivo = next_line
        arquivo = arquivo.replace("\n", "").replace(" ", "")

        arquivo2 = next_line2
        arquivo2 = arquivo2.replace("\n", "").replace(" ", "")

        value, variation = [x for x in arquivo.split(',')]

        strat = []
        strat.append('/Strategies/Script_PMLINST_WF.py') # PMLINST WF
        strat.append('/Strategies/Script_PMLINST_F1.py') # PMLINST F1
        strat.append('/Strategies/Script_PMLINST_F2.py') # PMLINST F2
        strat.append('/Strategies/Script_PMLALG_WF.py') # PMLALG WF
        strat.append('/Strategies/Script_PMLALG_F1.py') # PMLINST F1
        strat.append('/Strategies/Script_PMLALG_F2.py') # PMLINST F2        
        strat.append('/Strategies/Script_ProactiveOnline_PMLINST_WithoutFeedback - II.py') # PMLINST WF ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLINST_F1 - II.py') # PMLINST F1 ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLINST_F2 - II.py') # PMLINST F2 ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLALG_WithoutFeedback - II.py') # PMLALG WF ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLALG_F1 - II.py') # PMLINST F1 ProactiveOnline
        strat.append('/Strategies/Script_ProactiveOnline_PMLALG_F2 - II.py') # PMLINST F2 ProactiveOnline
        strat.append('/Strategies/Script_Corrective.py') # Corrective

        Best = Infinity

        value = int(float(value))
        variation = int(float(variation))

        if value == 4:
            value = 1
        else:
            if value == 6:
                value = 2
            else:
                if value == 8:
                    value = 3
                else:
                    if value == 10:
                        value = 4
                    else:
                        if value == 12:
                            value = 5
                        else:
                            if value == 15:
                                value = 6
                            else:
                                if value == 20:
                                    value = 7
                                else:
                                    if value == 50:
                                        value = 8
                                    else:
                                        value = 9

        for j in range(0, totalStrat):
            result = run_path(strat[j], init_globals={"value": value, "variation": variation})
            result = result.get('tardiness')
            K.clear_session()

            print("Result:", result,"Strategy:", j, "Cluster:", i)
  
            if result < Best:
                Best = result
                Best_Strategy = j       

        Rest.write(str(arquivo2))
        Rest.write(",%s"%str(round(Best,2)))
        Rest.write(",%s\n"%str(round(Best_Strategy,2)))
        
        next_line = myfile.readline()
        next_line2 = myfile2.readline()
        