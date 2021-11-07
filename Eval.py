import pickle
from dateutil.parser import parse

def get_available_products():
    return ["Mint Sauce", "Pilau Rice", "Curry", "Chicken Tikka Masala", "Saag Aloo", "Mixed Starter", "Egg Rice"]

def date_parser(dates):
    dates = [x.date() for x in dates]
    return sorted(list(set(dates)))
def transform_data(sequence, preddictions ,count):
    x = sequence + preddictions
    return x[count:]
def read_data(prod):
    data = {}
    with (open(prod + ".pkl", "rb")) as openfile:
        while True:
            try:
                data = pickle.load(openfile)
            except EOFError:
                break
    orig = data['orig'].tolist()
    predicted = data['predicted'].tolist()
    date = date_parser(data['date'].tolist())
    date_tuple = []
    for (day, idx) in zip(date, range(len(date))):
        date_tuple.append((orig[idx][0], predicted[idx][0], day))
    return date_tuple
def get_predictions(date, count, prod):
    ret = {
        'orig':[],
        'pred':[],
        'date':[]
    }
    data = read_data(prod)
    idx = 0
    for x in range(len(data)):
        if(data[x][2] == date):
            idx = x

    ret['orig'] = [x[0] for x in data[idx:idx + count]]
    ret['pred'] = [x[1] for x in data[idx:idx + count]]
    ret['date'] = [x[2] for x in data[idx:idx + count]]
    return  ret

def get_suggestions():
    with (open("menus.pkl", "rb")) as openfile:
        while True:
            try:
                data = pickle.load(openfile)
            except EOFError:
                break
    return data


if __name__=="__main__":
    print(get_predictions(parse("2018-04-21T").date(), 10, "data_processed"))
    print(get_suggestions())