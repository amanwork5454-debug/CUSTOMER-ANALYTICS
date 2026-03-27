import pickle
with open('model.pkl', 'rb') as f:
    m = pickle.load(f)
print(type(m))
if isinstance(m, dict):
    print("Keys:", m.keys())
else:
    print("Not a dict - old format!")