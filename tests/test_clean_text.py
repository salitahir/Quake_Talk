# tests/test_clean_text.py
from quake_talk.preprocessing.clean_text import clean_tweet


def test_empty_and_none():
    assert clean_tweet("") == ""
    assert clean_tweet(None) == ""


def test_remove_mentions_urls_and_stopwords():
    inp = "Hello @user! Visit http://example.com. This is GREAT!!!"
    out = clean_tweet(inp)
    assert "hello" in out
    assert "user" not in out
    assert "visit" in out      # 'visit' stays, since it's not in default stopwords
    # ensure stopwords are removed at the token level, not substring
    tokens = out.split()
    assert "this" not in tokens
    assert "is"   not in tokens


def test_punctuation_stripped():
    inp = "Wow!!! So good: awesome, amazing;"
    out = clean_tweet(inp, remove_stopwords=False)
    assert all(ch not in out for ch in "!,:;" )
