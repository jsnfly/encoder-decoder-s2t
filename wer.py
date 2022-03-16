import jiwer

def calculate_wer(predictions, golds):

    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])
    return jiwer.wer(golds, predictions, truth_transform=transformation, hypothesis_transform=transformation)