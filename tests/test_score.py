from answerable.judge import INVALID, OK, UNANSWERED, UNGROUNDED, Verdict
from answerable.score import summarise


def v(status, question="¿?"):
    return Verdict(question, status, answer="a", evidence="e")


def test_the_score_is_the_share_of_questions_provably_answered():
    assert summarise([v(OK), v(OK), v(OK), v(UNANSWERED)]).value == 75


def test_an_unprovable_answer_counts_as_a_miss():
    # A buyer reading the page would not have found it either.
    assert summarise([v(OK), v(UNGROUNDED)]).value == 50


def test_an_unreadable_reply_counts_as_a_miss():
    assert summarise([v(OK), v(INVALID)]).value == 50


def test_no_questions_is_zero_not_a_crash():
    assert summarise([]).value == 0


def test_every_kind_of_miss_is_listed_as_a_gap():
    s = summarise([v(OK, "a"), v(UNANSWERED, "b"), v(UNGROUNDED, "c"), v(INVALID, "d")])
    assert s.gaps == ["b", "c", "d"]
    assert s.ok == 1 and s.unanswered == 1 and s.ungrounded == 1 and s.invalid == 1


def test_only_the_invented_answers_are_flagged_as_hallucinations():
    s = summarise([v(OK, "a"), v(UNGROUNDED, "c"), v(UNANSWERED, "b")])
    assert [h.question for h in s.hallucinations] == ["c"]


def test_grounding_is_the_share_of_claims_that_could_be_proved():
    assert summarise([v(OK), v(OK), v(OK), v(UNGROUNDED)]).grounding == 75


def test_grounding_is_unknown_when_nothing_was_claimed():
    # 0 of 0 is not 0%: printing 0% would accuse an honest run of lying.
    assert summarise([v(UNANSWERED), v(UNANSWERED)]).grounding is None


def test_bands_read_the_way_a_client_expects():
    assert summarise([v(OK)] * 10).band == "excelente"
    assert summarise([v(OK)] * 8 + [v(UNANSWERED)] * 2).band == "bueno"
    assert summarise([v(OK)] * 6 + [v(UNANSWERED)] * 4).band == "flojo"
    assert summarise([v(UNANSWERED)] * 10).band == "critico"


def test_every_score_lands_in_exactly_one_band():
    for ok in range(0, 21):
        s = summarise([v(OK)] * ok + [v(UNANSWERED)] * (20 - ok))
        assert s.band in ("excelente", "bueno", "flojo", "critico")
        assert s.verdict_line
