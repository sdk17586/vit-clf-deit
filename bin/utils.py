def calculateAccuracy(yPred, label):
    
    topPred = yPred.argmax(1, keepdim=True)
    correct = topPred.eq(label.view_as(topPred)).sum()
    accuracy = correct.float() / label.shape[0]

    return accuracy

