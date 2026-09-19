"""Tests for sleepyjean.memory_store.MemoryStore."""

from tests.test_sleepyjean.conftest import rand_emb


class TestMemoryStore:
    def test_add_episode_and_search(self, store):
        """add_episode + search で格納と検索が動作する"""
        emb = rand_emb()
        eid = store.add_episode(emb, {"source": "dialogue"})
        assert eid

        results = store.search(emb, n_results=1)
        assert len(results) == 1
        assert results[0]["episode_id"] == eid

    def test_add_abstract_included_in_search(self, store):
        """add_abstract + search(include_abstract=True) で抽象記憶が検索対象に含まれる"""
        emb = rand_emb()
        aid = store.add_abstract(emb, cluster_id="c1")

        results = store.search(emb, n_results=1, include_abstract=True)
        assert len(results) == 1
        assert results[0]["episode_id"] == aid

    def test_abstract_excluded_from_search(self, store):
        """search(include_abstract=False) で抽象記憶が除外される"""
        emb_abstract = rand_emb()
        store.add_abstract(emb_abstract, cluster_id="c1")

        emb_episode = rand_emb()
        store.add_episode(emb_episode)

        results = store.search(emb_abstract, n_results=10, include_abstract=False)
        for r in results:
            assert r["metadata"].get("is_abstract") is not True

    def test_delete(self, store):
        """delete で指定エピソードが削除される"""
        emb = rand_emb()
        eid = store.add_episode(emb)

        deleted = store.delete([eid])
        assert deleted == 1

        results = store.search(emb, n_results=1)
        assert len(results) == 0

    def test_replace_abstracts(self, store):
        """replace_abstracts で既存クラスタ重心が全置換される"""
        emb1 = rand_emb()
        store.add_abstract(emb1, cluster_id="old_c1")

        new_emb = rand_emb()
        store.replace_abstracts([
            {"embedding": new_emb, "cluster_id": "new_c1"},
        ])

        stats = store.stats()
        assert stats["abstract_count"] == 1
        assert stats["cluster_count"] == 1

    def test_search_increments_access_count(self, store):
        """search が access_count をインクリメントする"""
        emb = rand_emb()
        eid = store.add_episode(emb)

        store.search(emb, n_results=1)
        store.search(emb, n_results=1)

        row = store._db.execute(
            "SELECT access_count FROM memory_metadata WHERE episode_id = ?",
            (eid,),
        ).fetchone()
        assert row["access_count"] == 2

    def test_stats(self, store):
        """stats が正しい値を返す"""
        store.add_episode(rand_emb())
        store.add_episode(rand_emb())
        store.add_abstract(rand_emb(), cluster_id="c1")

        stats = store.stats()
        assert stats["total"] == 3
        assert stats["episode_count"] == 2
        assert stats["abstract_count"] == 1
        assert stats["cluster_count"] == 1
        assert stats["last_updated"] is not None

    def test_search_empty_store(self, store):
        """空のストアに対する search が空リストを返す"""
        results = store.search(rand_emb(), n_results=3)
        assert results == []
