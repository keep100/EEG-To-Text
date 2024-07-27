import pickle
import random


def construct_augmented_data(
    rawData=None,
    factor=0.1,
    bands=["GD_t1", "GD_t2", "GD_a1", "GD_a2", "GD_b1", "GD_b2", "GD_g1", "GD_g2"],
):
    subjects = list(rawData.keys())
    train_divider=int(0.8*len(rawData[subjects[0]]))
    for subject in subjects:
        random_list = subjects[:]
        random_list.remove(subject)
        random_subject = random.sample(random_list, 1)[0]

        for i in range(train_divider):
            sentence_obj = rawData[subject][i]
            random_sentence_obj = rawData[random_subject][i]
            if sentence_obj is None or random_sentence_obj is None:
                continue
            intersection_word = set(sentence_obj["word_tokens_has_fixation"]) & set(random_sentence_obj["word_tokens_has_fixation"])
            for word_obj in sentence_obj["word"]:
                if word_obj["content"] in intersection_word:
                    for random_word_obj in random_sentence_obj["word"]:
                        if word_obj["content"] == random_word_obj["content"]:
                            for band in bands:
                                if len(word_obj["word_level_EEG"]["GD"][band]) and \
                                    len(word_obj["word_level_EEG"]["GD"][band]) == len(random_word_obj["word_level_EEG"]["GD"][band]):
                                    word_obj["word_level_EEG"]["GD"][band] = (1 - factor) * word_obj["word_level_EEG"]["GD"][band] + factor * random_word_obj["word_level_EEG"]["GD"][band]
                            break
    return rawData


# ZAB (sentence_level_EEG word word_tokens_has_fixation)   word_level_EEG GD

if __name__ == '__main__':
    dataset_path_task1 = "./dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle"
    with open(dataset_path_task1, "rb") as handle:
        task1 = pickle.load(handle)
    print(construct_augmented_data(rawData=task1, factor=0.1).keys())
