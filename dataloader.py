import dgl
import torch
import pickle
import os.path as osp
import dgl.function as dfn


def load_data(dataset='TMDB'):
    dataset = dataset.lower()
    assert dataset in ['tmdb', 'croval', 'arxiv', 'book', 'dblp', 'patent']

    data_path = osp.join('data', dataset, dataset + '.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if dataset == 'tmdb':
        target = 'movie'
        movie_actor_mid = torch.from_numpy(data['movie-actor'][0])
        movie_actor_aid = torch.from_numpy(data['movie-actor'][1])
        movie_director_mid = torch.from_numpy(data['movie-director'][0])
        movie_director_did = torch.from_numpy(data['movie-director'][1])
        edge_dict = {
            ('actor', 'performs', 'movie'): (movie_actor_aid, movie_actor_mid),
            ('movie', 'performed_by', 'actor'): (movie_actor_mid, movie_actor_aid),
            ('director', 'directs', 'movie'): (movie_director_did, movie_director_mid),
            ('movie', 'directed_by', 'director'): (movie_director_mid, movie_director_did),
        }
        g = dgl.heterograph(edge_dict, idtype=torch.int64)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)

        # load movie features
        g.nodes['movie'].data['feat'] = torch.from_numpy(data['movie_feats'])

        # generate actor/director features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='performed_by')
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='directed_by')
            return g

        labels = torch.tensor(data['movie_labels'], dtype=torch.int64)
        g.nodes['movie'].data['label'] = labels
        g.target = target

        years = data['movie_years']
        year_splits = [2015, 2016, 2018, 2019]
    elif dataset == 'croval':
        target = 'question'
        question_src_id = torch.tensor(data['question-question'][0])
        question_dst_id = torch.tensor(data['question-question'][1])
        question_user_qid = torch.tensor(data['question-user'][0])
        question_user_uid = torch.tensor(data['question-user'][1])
        question_tag_qid = torch.tensor(data['question-tag'][0])
        question_tag_tid = torch.tensor(data['question-tag'][1])

        edge_dict = {
            ('question', 'links', 'question'): (torch.cat([question_src_id, question_dst_id]),
                                                torch.cat([question_dst_id, question_src_id])),
            ('user', 'asks', 'question'): (question_user_uid, question_user_qid),
            ('question', 'asked_by', 'user'): (question_user_qid, question_user_uid),
            ('question', 'contains', 'tag'): (question_tag_qid, question_tag_tid),
            ('tag', 'contained_by', 'question'): (question_tag_tid, question_tag_qid),
        }
        g = dgl.heterograph(edge_dict, idtype=torch.int64)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)
        g = dgl.remove_self_loop(g, etype='links')
        g = dgl.add_self_loop(g, etype='links')

        # load question features
        g.nodes['question'].data['feat'] = torch.from_numpy(data['question_feats'])

        # generate user/tag features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='asked_by')
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='contains')
            return g

        labels = torch.tensor(data['question_labels'], dtype=torch.int64)
        g.nodes['question'].data['label'] = labels
        g.target = target

        years = data['question_years']
        year_splits = [2011, 2012, 2012, 2013]
    elif dataset == 'arxiv':
        target = 'paper'
        paper_src_id = torch.from_numpy(data['paper-paper'][0])
        paper_dst_id = torch.from_numpy(data['paper-paper'][1])
        paper_author_pid = torch.from_numpy(data['paper-author'][0])
        paper_author_aid = torch.from_numpy(data['paper-author'][1])
        paper_fos_pid = torch.from_numpy(data['paper-fos'][0])
        paper_fos_fid = torch.from_numpy(data['paper-fos'][1])
        edge_dict = {
            ('paper', 'cites', 'paper'): (torch.cat([paper_src_id, paper_dst_id]),
                                          torch.cat([paper_dst_id, paper_src_id])),
            ('author', 'writes', 'paper'): (paper_author_aid, paper_author_pid),
            ('paper', 'written_by', 'author'): (paper_author_pid, paper_author_aid),
            ('fos', 'topics', 'paper'): (paper_fos_fid, paper_fos_pid),
            ('paper', 'has_topic', 'fos'): (paper_fos_pid, paper_fos_fid)
        }
        g = dgl.heterograph(edge_dict, idtype=torch.int64)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)
        g = dgl.remove_self_loop(g, etype='cites')
        g = dgl.add_self_loop(g, etype='cites')

        # load paper features
        g.nodes['paper'].data['feat'] = torch.from_numpy(data['paper_feats'])

        # generate author/fos features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='written_by')
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='has_topic')
            return g

        labels = torch.tensor(data['paper_labels'], dtype=torch.int64)
        g.nodes['paper'].data['label'] = labels
        g.target = target

        years = data['paper_years']
        year_splits = [2017, 2018, 2018, 2019]
    elif dataset == 'book':
        target = 'book'
        book_src_id = torch.from_numpy(data['book-book'][0])
        book_dst_id = torch.from_numpy(data['book-book'][1])
        book_author_bid = torch.from_numpy(data['book-author'][0])
        book_author_aid = torch.from_numpy(data['book-author'][1])
        book_publisher_bid = torch.from_numpy(data['book-publisher'][0])
        book_publisher_pid = torch.from_numpy(data['book-publisher'][1])
        edge_dict = {
            ('book', 'similar_to', 'book'): (torch.cat([book_src_id, book_dst_id]),
                                          torch.cat([book_dst_id, book_src_id])),
            ('author', 'writes', 'book'): (book_author_aid, book_author_bid),
            ('book', 'written_by', 'author'): (book_author_bid, book_author_aid),
            ('publisher', 'publishes', 'book'): (book_publisher_pid, book_publisher_bid),
            ('book', 'published_by', 'publisher'): (book_publisher_bid, book_publisher_pid)
        }
        g = dgl.heterograph(edge_dict, idtype=torch.int64)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)
        g = dgl.remove_self_loop(g, etype='similar_to')
        g = dgl.add_self_loop(g, etype='similar_to')

        # load book features
        g.nodes['book'].data['feat'] = torch.from_numpy(data['book_feats'])

        # generate author/publisher features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='written_by')
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='published_by')
            return g

        labels = torch.tensor(data['book_labels'].todense(), dtype=torch.float32)
        g.nodes['book'].data['label'] = labels
        g.target = target

        years = data['book_years']
        year_splits = [2011, 2012, 2012, 2013]
    elif dataset == 'dblp':
        target = 'paper'
        paper_src_id = torch.from_numpy(data['paper-paper'][0])
        paper_dst_id = torch.from_numpy(data['paper-paper'][1])
        paper_author_pid = torch.from_numpy(data['paper-author'][0])
        paper_author_aid = torch.from_numpy(data['paper-author'][1])
        paper_fos_pid = torch.from_numpy(data['paper-fos'][0])
        paper_fos_fid = torch.from_numpy(data['paper-fos'][1])
        edge_dict = {
            ('paper', 'cites', 'paper'): (torch.cat([paper_src_id, paper_dst_id]),
                                          torch.cat([paper_dst_id, paper_src_id])),
            ('author', 'writes', 'paper'): (paper_author_aid, paper_author_pid),
            ('paper', 'written_by', 'author'): (paper_author_pid, paper_author_aid),
            ('fos', 'topics', 'paper'): (paper_fos_fid, paper_fos_pid),
            ('paper', 'has_topic', 'fos'): (paper_fos_pid, paper_fos_fid)
        }
        g = dgl.heterograph(edge_dict, idtype=torch.int64)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)
        g = dgl.remove_self_loop(g, etype='cites')
        g = dgl.add_self_loop(g, etype='cites')

        # load paper features
        g.nodes['paper'].data['feat'] = torch.from_numpy(data['paper_feats'])

        # generate author/fos features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='written_by')
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='has_topic')
            return g

        labels = torch.tensor(data['paper_labels'], dtype=torch.int64)
        g.nodes['paper'].data['label'] = labels
        g.target = target

        years = data['paper_years']
        # year_splits = [2015, 2016, 2017, 2018]
        # year_splits = [2015, 2016, 2016, 2017]
        year_splits = [2010, 2011, 2013, 2014]
    elif dataset == 'patent':
        target = 'patent'
        patent_inventor_pid = torch.from_numpy(data['patent-inventor'][0])
        patent_inventor_iid = torch.from_numpy(data['patent-inventor'][1])
        patent_examiner_pid = torch.from_numpy(data['patent-examiner'][0])
        patent_examiner_eid = torch.from_numpy(data['patent-examiner'][1])
        edge_dict = {
            ('inventor', 'invents', 'patent'): (patent_inventor_iid, patent_inventor_pid),
            ('patent', 'invented_by', 'inventor'): (patent_inventor_pid, patent_inventor_iid),
            ('examiner', 'examines', 'patent'): (patent_examiner_eid, patent_examiner_pid),
            ('patent', 'examined_by', 'examiner'): (patent_examiner_pid, patent_examiner_eid),
        }
        g = dgl.heterograph(edge_dict, idtype=torch.int64)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)

        # load patent features
        g.nodes['patent'].data['feat'] = torch.from_numpy(data['patent_feats'])

        # generate inventor/examiner features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='invented_by')
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='examined_by')
            return g

        labels = torch.tensor(data['patent_labels'], dtype=torch.int64)
        g.nodes['patent'].data['label'] = labels
        g.target = target

        years = data['patent_years']
        year_splits = [2014, 2015, 2015, 2016]

    idx_train = torch.from_numpy((years <= year_splits[0]).nonzero()[0])
    idx_val = torch.from_numpy(((years >= year_splits[1]) &
                                (years <= year_splits[2])).nonzero()[0])
    idx_test = torch.from_numpy((years >= year_splits[3]).nonzero()[0])

    print(g)
    print((idx_train.shape[0], idx_val.shape[0], idx_test.shape[0]))
    return g, (idx_train, idx_val, idx_test), generate_node_features
