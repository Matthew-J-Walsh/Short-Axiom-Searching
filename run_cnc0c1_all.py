from BooleanAlgebraMethod import GenerateTautologyFile
import FileUtilities
from VampireHandler import Run_Vampire_Elimination

def run():
    GenerateTautologyFile(17, "C017Results/C0_17_Full", "C0")
    GenerateTautologyFile(19, "C019Results/C0_19_Full", "C0")
    GenerateTautologyFile(17, "C117Results/C1_17_Full", "C1")
    GenerateTautologyFile(19, "C119Results/C1_19_Full", "C1")
    GenerateTautologyFile(17, "CN20Results/CN_20_Full", "CN")
    GenerateTautologyFile(19, "CN21Results/CN_21_Full", "CN")
    
    FileUtilities.CombineFiles("C017Results", "C0_17_Full", new_ending='')
    FileUtilities.CombineFiles("C019Results", "C0_19_Full", new_ending='')
    FileUtilities.CombineFiles("C117Results", "C1_17_Full", new_ending='')
    FileUtilities.CombineFiles("C119Results", "C1_19_Full", new_ending='')
    FileUtilities.CombineFiles("CN20Results", "CN_20_Full", new_ending='')
    FileUtilities.CombineFiles("CN21Results", "CN_21_Full", new_ending='')
    
    #Run_Vampire_Elimination("C017Results", "C0_17_Full.txt", [], "C1")
    #Run_Vampire_Elimination("C019Results", "C0_19_Full.txt", [], "C1")
    #Run_Vampire_Elimination("C117Results", "C1_17_Full.txt", [], "C1")
    #Run_Vampire_Elimination("C119Results", "C1_19_Full.txt", [], "C1")
    Run_Vampire_Elimination("CN20Results", "CN_20_Full.txt", ["i(i(i(i(i(X,Y),i(n(Z),n(U))),Z),V),i(i(V,X),i(U,X)))"], "C1")
    Run_Vampire_Elimination("CN21Results", "CN_21_Full.txt", ["i(i(i(i(i(X,Y),i(n(Z),n(U))),Z),V),i(i(V,X),i(U,X)))"], "C1")
    
    