# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:47:33 2020

@author: Rupert.Thomas

Pytest test cases for rank_results.py

"""

from numpy.testing import assert_almost_equal
from hypothesis import given
from hypothesis.strategies import text

from sklearn_tfidf_rank_results import rank_documents


def test_rank_documents1():
    """
    Test similarity match - positive control
    """
    
    # given
    search_terms = 'hello world'
    documents = [search_terms]

    # when
    result = rank_documents(search_terms, documents)

    # then
    assert_almost_equal(result, [1])



@given(search_terms=text())
def test_rank_documents2(search_terms):
    """
    Test similarity match - fuzzing
    """
    
    # given
    # search_terms = 'hello world'
    documents = ['the quick brown fox jumped over the lazy dog', 'Joe Jackson dances in Paris']

    # when
    result = rank_documents(search_terms, documents)

    # then
    assert type(result) is list
    for item in result:
        assert type(item) is float


@given(search_terms=text(), document1=text(), document2=text())
def test_rank_documents2(search_terms, document1, document2):
    """
    Test similarity match - fuzzing
    """
    
    # given
    # search_terms = 'hello world'
    documents = [document1, document2]

    # when
    result = rank_documents(search_terms, documents)

    # then
    assert type(result) is list
    for item in result:
        assert type(item) is float
