from router import RuleRouter, rule_score


def test_easy_question_scores_low():
    assert rule_score("What is a compiler?") == 0.0


def test_code_and_hard_word_score_high():
    assert rule_score("Why does this crash? int *p; *p = 10;") >= 0.7


def test_long_question_adds_score():
    assert rule_score(" ".join(["word"] * 50)) == 0.4


def test_score_is_capped_at_one():
    q = "Why " + "debug this code { x = 1 + 2; } " * 20
    assert rule_score(q) == 1.0


def test_rule_router_batch_matches_single():
    r = RuleRouter()
    qs = ["What is a stack?", "Prove that n^2 >= n", "Trace: for(i=0;i<3;i++);"]
    assert list(r.p_hard_many(qs)) == [r.p_hard(q) for q in qs]


def test_cheapest_threshold_picks_cheapest_setting_that_meets_target():
    from router import cheapest_threshold
    # 4 questions: the small model fails the two "hard" ones (high scores).
    scores = [0.1, 0.2, 0.7, 0.9]
    small_ok, large_ok = [1, 1, 0, 0], [1, 1, 1, 1]
    small_cost, large_cost = [0, 0, 0, 0], [1, 1, 1, 1]
    t, acc, cost = cheapest_threshold(scores, small_ok, large_ok, small_cost, large_cost,
                                      [0.0, 0.5, 0.8, 1.01], target=0.95)
    assert (t, acc, cost) == (0.5, 1.0, 0.5)   # only the two hard ones go to large


def test_cheapest_threshold_cascade_counts_both_calls():
    from router import cheapest_threshold
    conf = [9, 9, 3, 2]
    t, acc, cost = cheapest_threshold(conf, [1, 1, 0, 0], [1, 1, 1, 1], [0.1] * 4, [1] * 4,
                                      range(1, 12), target=0.95, cascade=True)
    assert t == 4 and acc == 1.0 and abs(cost - (0.1 + 0.5)) < 1e-9
