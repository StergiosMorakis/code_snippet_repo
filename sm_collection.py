def get_context(
    text: str,
    start_pos: int,
    end_pos: int,
    token_window: int = 100,
    validation_token_length = 4,
    tokenizer: object = None,       # MUST be set 
    #   ( e.g. nltk.tokenize.RegexpTokenizer(r"\w+").tokenize(text) )
) -> tuple:
    '''
        Desc:
            Get left and right context (token_window / 2 length per side)
            surrounding mention m (text[start_pos : end_pos + 1])
        Note: 
            Variable end_pos is the index of the mention's last char in text.
            (Some packages (e.g. spaCy) output end_pos+1)
        Note 2: 
            from nltk.tokenize import RegexpTokenizer
            tokenizer = RegexpTokenizer(r"\w+")
        Return:
            tuple(left_context, right_context)
    '''

    def get_left_context(
        text: str,
        token_window: int,
        validation_token_length:  int = 4,
        tokenizer: object,
    ) -> str:
        tokens = tokenizer.tokenize(text)
        if len(tokens) <= token_window:
            return text
        valid_index = len(tokens) - token_window
        validation_tokens = tokens[valid_index : valid_index + validation_token_length]
        tmp_index = 0
        cached_index_pos = []
        for i, token in enumerate(tokens):
            tmp_index = text.find(token, tmp_index)
            if (
                token == validation_tokens[-1]
                and tokens[i-validation_token_length+1:i] == validation_tokens[:-1]
            ):
                return text[cached_index_pos[-validation_token_length+1]:]
            cached_index_pos.append(tmp_index)
            tmp_index = tmp_index + len(token)
        return text

    def get_right_context(
        text: str,
        token_window: int,
        validation_token_length: int = 4,
        tokenizer: object,
    ) -> str:
        tokens = tokenizer.tokenize(text)
        if len(tokens) <= token_window:
            return text
        validation_tokens = tokens[token_window - validation_token_length + 1 : token_window + 1]
        tmp_index = 0
        for i, token in enumerate(tokens):
            if (
                token == validation_tokens[-1]
                and tokens[i-validation_token_length+1:i] == validation_tokens[:-1]
            ):
                return text[:tmp_index]
            tmp_index = text.find(token, tmp_index)
            tmp_index = tmp_index + len(token)
        return text
    
    assert tokenizer, 'Tokenizer is None (cannot retrieve context).'
    assert start_pos <= end_pos, 'Mention indices do not satisfy requirements.'
    left_context = get_left_context(
        text[:start_pos],
        token_window // 2,
        validation_token_length,
        tokenizer,
    )
    right_context = get_right_context(
        text[end_pos+1:],
        token_window // 2,
        validation_token_length,
        tokenizer,
    )
    return (
        left_context,
        right_context
    )

 def compute_precision_recall_f1score(
    ground_truth_spans: set,
    detected_spans: set, 
    verbose: bool = False,
) -> tuple:
    '''
        Soft eval:
            True Positive is 
                a ground truth span for which
                at least one detected span
                existed within its boundaries.
        returns:
            tuple(precision, recall, f1_score)
    '''

    ###
    # ground_truth_spans: set= {(1, 3), (6,9), (11, 14), (19, 33)}  
    # detected_spans: set = {(1, 3), (10, 12), (12, 13), (13,14), (100, 101), (200,201)}
    ###

    def compute_precision(tp: int, fp: int) -> float:
        return tp / (tp + fp)

    def compute_recall(tp: int, fn: int) -> float:
        return tp / (tp + fn)

    def compute_f1score(precision: float, recall: float) -> float:
        return 2 * precision * recall / (precision + recall)

    def print_stats(*data) -> None:
        tp, fp, fn, precision, recall, f1_score = data
        print('True positives',     tp)
        print('False positives',    fp)
        print('False negatives',    fn)
        print('Precision',          precision)
        print('Recall',             recall)
        print('F1-score',           f1_score)

    detected_ground_truth_spans = set()
    for detected_span in detected_spans:
        for ground_truth_span in ground_truth_spans:
            if (
                detected_span[0] >= ground_truth_span[0] 
                and detected_span[1] <= ground_truth_span[1]
            ):
                detected_ground_truth_spans.add(ground_truth_span)
                break
        else:
            detected_ground_truth_spans.add(detected_span)
    tp, fp, fn = (
        len(
            detected_ground_truth_spans.intersection(ground_truth_spans)
        ),
        len(
            detected_ground_truth_spans.difference(ground_truth_spans)
        ),
        len(
            ground_truth_spans.difference(detected_ground_truth_spans)
        )
    )
    precision = compute_precision(tp, fp)
    recall = compute_precision(tp, fn)
    f1_score = compute_f1score(precision, recall)
    if verbose:
        print_stats(
            tp, fp, fn, precision, recall, f1_score
        )
    return precision, recall, f1_score

if __name__ == '__main__':
    pass