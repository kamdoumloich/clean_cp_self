#
# FIlters for interesting cases
#
class LargeDeviationInterestingCaseFilter:
    def __init__(self,probabilityOfInterest):
        self.probabilityOfInterest = probabilityOfInterest
        self.casesByPredictor = {}
        self.currentCases = []
        self.currentPredictor = None
        self.currentCalibrationParameter = None
        # self.calibrationParameter = []

    def resetFilter(self):
        self.casesByPredictor = {}
        self.currentCases = []
        self.currentPredictor = None
        self.currentCalibrationParameter = None

    def reportNewContext(self,predictorNumber,calibrationParameter):
        self.currentCases = []
        self.currentPredictor = predictorNumber
        self.currentCalibrationParameter = calibrationParameter

    def reportCase(self,dataPointNum,probabilities,realClass,predicted):
        self.currentCases.append((dataPointNum,probabilities,realClass,predicted))

    def reportProbabilityOfBeingCorrect(self,probability):
        # Store the results away for the first time the threshold probability is crossed
        if probability>=self.probabilityOfInterest:
            if not self.currentPredictor in self.casesByPredictor:
                self.casesByPredictor[self.currentPredictor] = (probability,self.currentCalibrationParameter,self.currentCases.copy())

    def getCalibrationParameter(self, predictorNumber):
        # if self.casesByPredictor.get(predictorNumber, -1) == -1:
        if predictorNumber not in self.casesByPredictor:
            raise NotImplementedError
        else:
            return self.casesByPredictor[predictorNumber][1]
        # return self.calibrationParameter

    def getTexReport(self):
        allLines = []
        allLines.append("\\textbf{Cases filtered:} Interesting cases with large deviation for target probability "+str(self.probabilityOfInterest))
        allLines.append("\n\n\\textbf{Parameter/Probability Combinations considered:}\\begin{enumerate}")
        predictorKeys = []
        for (a,b) in self.casesByPredictor.items():
            # self.calibrationParameter.append(b[1])
            allLines.append("\\item Calibration Value: "+str(b[1])+" with probability "+str(b[0]))
            nofItems = len(b[2])
            predictorKeys.append(a)
        allLines.append("\\end{enumerate}")
        for predA in range(0,len(predictorKeys)):
            for predB in range(predA+1,len(predictorKeys)):
                allLines.append("\n\n\\textbf{Big differences between predictors "+str(predA)+" and "+str(predB)+":}\\begin{enumerate}")

                differences = []
                for i in range(0,nofItems):
                    setA = self.casesByPredictor[predictorKeys[predA]][2][i][3]
                    setB = self.casesByPredictor[predictorKeys[predB]][2][i][3]
                    B = len(set(setA).intersection(set(setB)))
                    nofDiff = len(setA)+len(setB)-2*B
                    differences.append((nofDiff,i))
                differences.sort()
                del i
                for (difference,index) in differences[-5:]:
                    allLines.append("\\item Case no.~"+str(index)+" with real class "+str(self.casesByPredictor[predictorKeys[predA]][2][index][2]))
                    allLines.append("\\begin{itemize}")
                    for p in self.casesByPredictor[predictorKeys[predA]][2][index][1]:
                        allLines.append("\\item Prob.Dist.: "+str(p))
                    for predC in range(0,len(predictorKeys)):
                        allLines.append("\\item Conf.Pred."+str(predC)+" result: "+str(self.casesByPredictor[predictorKeys[predC]][2][index][3]))
                    allLines.append("\\end{itemize}")
                allLines.append("\\end{enumerate}")
        

        # allLines.append("\\end{enumerate}")



        return "\n".join(allLines)
