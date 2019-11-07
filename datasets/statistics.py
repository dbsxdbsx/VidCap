from tqdm import tqdm

from utils.text import extract_nouns_verbs, parse


def get_stats(dataset, top=40, style='print'):
    """
    Get a string of dataset statistics useful for printinf, or latex or csv entry
    
    :param dataset: the dataset to get the statistics of
    :param top: 
    :param style: print, latex, or csv the style we want the results in
    :return: a string of the stats
    """
    assert style in ['print', 'latex', 'csv']

    # get the stats to print
    boxes_p_cls, boxes_p_img, samples_p_cls = box_counts(dataset)

    if len(boxes_p_cls) == 0:
        output_str = "# Images: %d\n" \
              "# Boxes: %d\n" \
              "# Categories: %d\n" % \
              (len(dataset.sample_ids), 0, 0)
    else:
        output_str = "# Images: %d\n" \
              "# Boxes: %d\n" \
              "# Categories: %d\n" \
              "Boxes per image (min, avg, max): %d, %d, %d\n" \
              "Boxes per category (min, avg, max): %d, %d, %d\n\n\n" % \
              (len(dataset.sample_ids), sum(boxes_p_img), len(boxes_p_cls),
               min(boxes_p_img), sum(boxes_p_img) / len(boxes_p_img), max(boxes_p_img),
               min(boxes_p_cls), sum(boxes_p_cls) / len(boxes_p_cls), max(boxes_p_cls))

    boxes_p_cls_dict = {}
    for i in range(len(boxes_p_cls)):
        boxes_p_cls_dict[i] = boxes_p_cls[i]

    output_str += "Object Counts:\n"
    c = 0
    for key, value in sorted(((value, key) for (key, value) in boxes_p_cls_dict.items()), reverse=True):
        c += 1
        if c > top:
            break

        if style == "latex":
            if c % 2 == 0:
                output_str += "\\rowcolor{lightGrey}\n"
            output_str += "%s & %s \\\\\n" % (dataset.classes[value], key)
        elif style == "csv":
            output_str += "%s\t%s\n" % (dataset.classes[value], key)
        else:
            output_str += "%s: %s\n" % (dataset.classes[value], key)

    if dataset.captions:
        sents_p_img, words_p_img, vocab_p_img, imgs_with_word, caps_with_word, word_freqs, vocab_size = \
            caption_counts(dataset)

        output_str += "\n\n\n\n# Captions: %d\n" \
                      "# Words: %d\n" \
                      "# Nouns %d (%d%% of words)\n" \
                      "# Verbs %d (%d%% of words)\n" \
                      "Vocab: %d\n" \
                      "Nouns Vocab %d (%d%% of Vocab)\n" \
                      "Verbs Vocab %d (%d%% of Vocab)\n\n" \
                      "Captions per image (min, avg, max): %d, %d, %d\n" \
                      "Words per image (min, avg, max): %d, %d, %d\n" \
                      "Nouns per image (min, avg, max): %d, %d, %d\n" \
                      "Verbs per image (min, avg, max): %d, %d, %d\n\n" \
                      "Vocab (unique words) per image (min, avg, max): %d, %d, %d\n" \
                      "Nouns Vocab (unique words) per image (min, avg, max): %d, %d, %d\n" \
                      "Verbs Vocab (unique words) per image (min, avg, max): %d, %d, %d\n" % \
                      (sum(sents_p_img),
                       sum(words_p_img[0]),
                       sum(words_p_img[1]), int(100 * sum(words_p_img[1]) / float(sum(words_p_img[0]))),
                       sum(words_p_img[2]), int(100 * sum(words_p_img[2]) / float(sum(words_p_img[0]))),
                       vocab_size,
                       len(word_freqs[1].keys()),
                       int(100 * len(word_freqs[1].keys()) / float(len(word_freqs[0].keys()))),
                       len(word_freqs[2].keys()),
                       int(100 * len(word_freqs[2].keys()) / float(len(word_freqs[0].keys()))),
                       min(sents_p_img), sum(sents_p_img) / len(sents_p_img), max(sents_p_img),
                       min(words_p_img[0]), sum(words_p_img[0]) / len(words_p_img[0]), max(words_p_img[0]),
                       min(words_p_img[1]), sum(words_p_img[1]) / len(words_p_img[1]), max(words_p_img[1]),
                       min(words_p_img[2]), sum(words_p_img[2]) / len(words_p_img[2]), max(words_p_img[2]),
                       min(vocab_p_img[0]), sum(vocab_p_img[0]) / len(vocab_p_img[0]), max(vocab_p_img[0]),
                       min(vocab_p_img[1]), sum(vocab_p_img[1]) / len(vocab_p_img[1]), max(vocab_p_img[1]),
                       min(vocab_p_img[2]), sum(vocab_p_img[2]) / len(vocab_p_img[2]), max(vocab_p_img[2]))

        output_str += "\n\nImages containing word:\n"
        c = 0
        for key, value in sorted(((value, key) for (key, value) in imgs_with_word[0].items()), reverse=True):
            c += 1
            if c > top:
                break

            if style == "latex":
                if c % 2 == 0:
                    output_str += "\\rowcolor{lightGrey}\n"
                output_str += "%s & %s \\\\\n" % (value, key)
            elif style == "csv":
                output_str += "%s\t%s\n" % (value, key)
            else:
                output_str += "%s: %s\n" % (value, key)

        output_str += "\n\nImages containing noun:\n"
        c = 0
        for key, value in sorted(((value, key) for (key, value) in imgs_with_word[1].items()), reverse=True):
            c += 1
            if c > top:
                break

            if style == "latex":
                if c % 2 == 0:
                    output_str += "\\rowcolor{lightGrey}\n"
                output_str += "%s & %s \\\\\n" % (value, key)
            elif style == "csv":
                output_str += "%s\t%s\n" % (value, key)
            else:
                output_str += "%s: %s\n" % (value, key)

        output_str += "\n\nImages containing verb:\n"
        c = 0
        for key, value in sorted(((value, key) for (key, value) in imgs_with_word[2].items()), reverse=True):
            c += 1
            if c > top:
                break

            if style == "latex":
                if c % 2 == 0:
                    output_str += "\\rowcolor{lightGrey}\n"
                output_str += "%s & %s \\\\\n" % (value, key)
            elif style == "csv":
                output_str += "%s\t%s\n" % (value, key)
            else:
                output_str += "%s: %s\n" % (value, key)

        output_str += "\n\nCaptions containing word:\n"
        c = 0
        for key, value in sorted(((value, key) for (key, value) in caps_with_word[0].items()), reverse=True):
            c += 1
            if c > top:
                break

            if style == "latex":
                if c % 2 == 0:
                    output_str += "\\rowcolor{lightGrey}\n"
                output_str += "%s & %s \\\\\n" % (value, key)
            elif style == "csv":
                output_str += "%s\t%s\n" % (value, key)
            else:
                output_str += "%s: %s\n" % (value, key)

        output_str += "\n\nCaptions containing noun:\n"
        c = 0
        for key, value in sorted(((value, key) for (key, value) in caps_with_word[1].items()), reverse=True):
            c += 1
            if c > top:
                break

            if style == "latex":
                if c % 2 == 0:
                    output_str += "\\rowcolor{lightGrey}\n"
                output_str += "%s & %s \\\\\n" % (value, key)
            elif style == "csv":
                output_str += "%s\t%s\n" % (value, key)
            else:
                output_str += "%s: %s\n" % (value, key)

        output_str += "\n\nCaptions containing verb:\n"
        c = 0
        for key, value in sorted(((value, key) for (key, value) in caps_with_word[2].items()), reverse=True):
            c += 1
            if c > top:
                break

            if style == "latex":
                if c % 2 == 0:
                    output_str += "\\rowcolor{lightGrey}\n"
                output_str += "%s & %s \\\\\n" % (value, key)
            elif style == "csv":
                output_str += "%s\t%s\n" % (value, key)
            else:
                output_str += "%s: %s\n" % (value, key)

        output_str += "\n\nWord Frequencies:\n"
        c = 0
        for key, value in sorted(((value, key) for (key, value) in word_freqs[0].items()), reverse=True):
            c += 1
            if c > top:
                break

            if style == "latex":
                if c % 2 == 0:
                    output_str += "\\rowcolor{lightGrey}\n"
                output_str += "%s & %s \\\\\n" % (value, key)
            elif style == "csv":
                output_str += "%s\t%s\n" % (value, key)
            else:
                output_str += "%s: %s\n" % (value, key)

        output_str += "\n\nNoun Frequencies:\n"
        c = 0
        for key, value in sorted(((value, key) for (key, value) in word_freqs[1].items()), reverse=True):
            c += 1
            if c > top:
                break

            if style == "latex":
                if c % 2 == 0:
                    output_str += "\\rowcolor{lightGrey}\n"
                output_str += "%s & %s \\\\\n" % (value, key)
            elif style == "csv":
                output_str += "%s\t%s\n" % (value, key)
            else:
                output_str += "%s: %s\n" % (value, key)

        output_str += "\n\nVerb Frequencies:\n"
        c = 0
        for key, value in sorted(((value, key) for (key, value) in word_freqs[2].items()), reverse=True):
            c += 1
            if c > top:
                break

            if style == "latex":
                if c % 2 == 0:
                    output_str += "\\rowcolor{lightGrey}\n"
                output_str += "%s & %s \\\\\n" % (value, key)
            elif style == "csv":
                output_str += "%s\t%s\n" % (value, key)
            else:
                output_str += "%s: %s\n" % (value, key)

        output_str += get_noun_coverage_str(dataset, boxes_p_cls, word_freqs[1], words_p_img[1], top, style)

        output_str += get_missing_noun_str(dataset, word_freqs[1], top, style)

    return output_str


def box_counts(dataset):
    """
    Calculate dataset bounding box count statistics

    :param dataset: the dataset
    :return: # boxes per class, # boxes per image, # samples per class
    """

    boxes_p_cls = [0] * len(dataset.classes)
    samples_p_cls = [0] * len(dataset.classes)
    boxes_p_img = []
    for sample_id in tqdm(dataset.sample_ids, desc="Processing Statistics for Box Data"):
        boxes_this_img = 0
        boxes = dataset.sample_boxes(sample_id)
        samples_p_cls_flag = [0] * len(dataset.classes)
        for box in boxes:
            boxes_p_cls[int(box[4])] += 1
            boxes_this_img += 1
            if samples_p_cls_flag[int(box[4])] == 0:
                samples_p_cls_flag[int(box[4])] = 1
                samples_p_cls[int(box[4])] += 1

        boxes_p_img.append(boxes_this_img)

    return boxes_p_cls, boxes_p_img, samples_p_cls


def caption_counts(dataset):
    """
    Calculate dataset caption count statistics

    :param dataset: the dataset
    :return: # captions per image, # words per image, # vocab per image, # images with word, # captions with word,
             word frequncies, vocabulary size
    """
    imgs_with_word = [{}, {}, {}]  # words, nouns, verbs  # number of images containing this word at least once
    caps_with_word = [{}, {}, {}]  # words, nouns, verbs  # number of captions containing this word at least once
    sents_p_img = []  # number of sentences per image
    words_p_img = [[], [], []]  # words, nouns, verbs  # the number of words per image
    vocab_p_img = [[], [], []]  # words, nouns, verbs  # the vocab size per image
    word_freqs = [{}, {}, {}]  # words, nouns, verbs  # the total word appearance counts

    for sample_id in tqdm(dataset.sample_ids, desc="Processing Statistics for Caption Data"):
        words_this_img = []
        nouns_this_img = []
        verbs_this_img = []
        captions = dataset.sample_captions(sample_id)

        for cap in captions:
            done_words = []
            words, nouns, verbs = extract_nouns_verbs(parse(cap), unique=False)
            for w in words:
                if w in word_freqs[0].keys():
                    word_freqs[0][w] += 1
                else:
                    word_freqs[0][w] = 1

                if w in nouns:
                    if w in word_freqs[1].keys():
                        word_freqs[1][w] += 1
                    else:
                        word_freqs[1][w] = 1

                if w in verbs:
                    if w in word_freqs[2].keys():
                        word_freqs[2][w] += 1
                    else:
                        word_freqs[2][w] = 1

                if w not in done_words:  # ensures we don't repeat a count for a repeated word in a caption
                    if w in caps_with_word[0].keys():
                        caps_with_word[0][w] += 1
                    else:
                        caps_with_word[0][w] = 1

                    if w in nouns:
                        if w in caps_with_word[1].keys():
                            caps_with_word[1][w] += 1
                        else:
                            caps_with_word[1][w] = 1

                    if w in verbs:
                        if w in caps_with_word[2].keys():
                            caps_with_word[2][w] += 1
                        else:
                            caps_with_word[2][w] = 1

                done_words.append(w)

            words_this_img += words
            nouns_this_img += nouns
            verbs_this_img += verbs

        sents_p_img.append(len(captions))

        words_p_img[0].append(len(words_this_img))
        words_p_img[1].append(len(nouns_this_img))
        words_p_img[2].append(len(verbs_this_img))

        vocab_this_img = set(words_this_img)
        nouns_vocab_this_img = set(nouns_this_img)
        verbs_vocab_this_img = set(verbs_this_img)

        vocab_p_img[0].append(len(vocab_this_img))
        vocab_p_img[1].append(len(nouns_vocab_this_img))
        vocab_p_img[2].append(len(verbs_vocab_this_img))

        for w in vocab_this_img:
            if w in imgs_with_word[0].keys():
                imgs_with_word[0][w] += 1
            else:
                imgs_with_word[0][w] = 1

            if w in nouns_this_img:
                if w in imgs_with_word[1].keys():
                    imgs_with_word[1][w] += 1
                else:
                    imgs_with_word[1][w] = 1

            if w in verbs_this_img:
                if w in imgs_with_word[2].keys():
                    imgs_with_word[2][w] += 1
                else:
                    imgs_with_word[2][w] = 1

    vocab_size = len(caps_with_word[0].keys())

    return sents_p_img, words_p_img, vocab_p_img, imgs_with_word, caps_with_word, word_freqs, vocab_size


def get_noun_coverage_str(dataset, boxes_p_cls, noun_freqs, nouns_p_img, top, style):
    overlaps = obj_noun_overlaps(dataset, boxes_p_cls, noun_freqs=noun_freqs)  # {obj: {count, nouns: {noun: count}}}

    output_str = "\n\n\nNoun Coverage:\n"
    c = 0
    top_nouns = 5
    tot_noun_occs = 0
    tot_perc = 0
    tot_obj_name_noun_count = 0
    for tot_count, obj_name in sorted(((value['total'], key) for (key, value) in overlaps.items()), reverse=True):
        c += 1
        if c > top:
            break

        noun_str = ""
        count_str = ""
        perc_str = ""
        cc = 0
        count_str_int = 0
        for count, noun in sorted(((count, noun) for (noun, count) in overlaps[obj_name]['synonyms'].items()),
                                  reverse=True):

            cc += 1
            count_str_int += count

            if cc == len(overlaps[obj_name]['synonyms'].keys()) and cc <= top_nouns:
                noun_str += "%s (%d)" % (noun, count)
            elif cc < top_nouns:
                noun_str += "%s (%d), " % (noun, count)
            elif cc == top_nouns:
                noun_str += "%s (%d), ... " % (noun, count)

        count_str += "%d" % count_str_int
        perc_str += "%0.2f" % ((100 * tot_count) / float(sum(nouns_p_img)))
        tot_noun_occs += count_str_int
        tot_perc += ((100 * tot_count) / float(sum(nouns_p_img)))

        # handle the noun with the exact obj_name spelling
        obj_name_noun_count = overlaps[obj_name]['exact']
        tot_obj_name_noun_count += obj_name_noun_count

        if style == "latex":
            if c % 2 == 0:
                output_str += "\\rowcolor{lightGrey}\n"
            output_str += "\\textbf{%s} & %d & %s & %s & %s \\\\\n" % (
                obj_name, obj_name_noun_count, noun_str, count_str, perc_str)
        elif style == "csv":
            output_str += "%s\t%d\t%s\t%s\t%s\n" % (obj_name, obj_name_noun_count, noun_str, count_str, perc_str)
        else:
            output_str += "%s: %d: %s: %s: %s\n" % (obj_name, obj_name_noun_count, noun_str, count_str, perc_str)

    if style == "latex":
        output_str += "\\rowcolor{grey8}\n"
        output_str += "\\textbf{TOTALS} & \\textbf{%d} & & \\textbf{%d} & \\textbf{%0.2f} \\\\\n" % (
            tot_obj_name_noun_count, tot_noun_occs, tot_perc)
    elif style == "csv":
        output_str += "TOTALS\t\t%d\t%d\t%0.2f\n" % (tot_obj_name_noun_count, tot_noun_occs, tot_perc)
    else:
        output_str += "TOTALS: : %d: %d: %0.2f\n" % (tot_obj_name_noun_count, tot_noun_occs, tot_perc)
    return output_str


def obj_noun_overlaps(dataset, boxes_p_cls, noun_freqs, use_synonyms=False):
    overlaps = {}
    for cls_idx, count in enumerate(boxes_p_cls):
        obj_name = dataset.classes[cls_idx]
        exact_count = 0
        total_count = 0

        synonym_counts = {}
        # make sure has syns, if doesn't go the else and just do the noun
        if use_synonyms and obj_name in dataset.obj_nouns().keys():
            for noun in dataset.obj_nouns()[obj_name]:
                if noun in noun_freqs.keys():
                    if noun == obj_name:
                        exact_count = noun_freqs[obj_name]
                    else:
                        synonym_counts[noun] = noun_freqs[noun]
                    total_count += noun_freqs[noun]
        else:
            if obj_name in noun_freqs.keys():
                exact_count = noun_freqs[obj_name]
                total_count = exact_count

        overlaps[obj_name] = {'count': count, 'exact': exact_count, 'synonyms': synonym_counts, 'total': total_count}

    return overlaps


def get_missing_noun_str(dataset, noun_freqs, top, style):
    groundings = []
    for v in dataset.obj_nouns().values():
        groundings += v

    output_str = "\n\n\nMissing Nouns:\n"
    c = 0
    for key, value in sorted(((value, key) for (key, value) in noun_freqs.items()), reverse=True):
        if value in groundings:
            # this exists as an object so don't count
            continue
        c += 1
        if c > top:
            break

        if style == "latex":
            if c % 2 == 0:
                output_str += "\\rowcolor{lightGrey}\n"
            output_str += "%s & %s \\\\\n" % (value, key)
        elif style == "csv":
            output_str += "%s\t%s\n" % (value, key)
        else:
            output_str += "%s: %s\n" % (value, key)
    return output_str
