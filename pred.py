from joblib import load
model = load('model.pkl')
input = "Giá dầu Brent tương_lai tăng 2,49 USD , tương_đương 7,3 % , lên 36,85 USD / thùng"
x = [str(input)]
pred = model.predict(x)
print(pred)