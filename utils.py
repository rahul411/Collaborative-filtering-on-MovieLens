from collections import defaultdict

def load_dataset(train_path, test_path, sep, user_based=True):
    all_users_set = set()
    all_movies_set = set()

    train_data = defaultdict(list)
    with open(train_path, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = map(lambda x: int(x),line.strip().split(sep))

            if user_based:
                train_data[uid].append((mid, float(rat)))
            else:
                train_data[mid].append((uid, float(rat)))

            if uid not in all_users_set:
                all_users_set.add(uid)
            if mid not in all_movies_set:
                all_movies_set.add(mid)

    tests = defaultdict(list)

    with open(test_path, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = map(lambda x: int(x),line.strip().split(sep))

            if user_based:
                tests[uid].append((mid, float(rat)))
            else:
                tests[mid].append((uid, float(rat)))

    return list(all_users_set), list(all_movies_set), train_data, tests