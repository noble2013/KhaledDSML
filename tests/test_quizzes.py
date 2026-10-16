import pytest
from khaled_dsml import get_questions, grade_quiz


def test_bank_and_grade():
    questions = get_questions("big_data", limit=8, seed=7)
    assert len(questions) == 8
    assert get_questions("big_data", limit=8, seed=7) == questions
    grade = grade_quiz(questions, {q.id: q.answer for q in questions})
    assert grade["percentage"] == 100 and grade["correct"] == 8


def test_wrong_answers():
    q = get_questions("data_science", limit=1)
    assert grade_quiz(q, {})["percentage"] == 0


def test_invalid_course():
    with pytest.raises(ValueError):
        get_questions("not-a-course")
