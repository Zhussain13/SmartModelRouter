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
