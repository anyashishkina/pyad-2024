# 1. Находим нужного пользователя
ratings = pd.read_csv('Ratings.csv')
zero_ratings = ratings[ratings['Book-Rating'] == 0]
user_with_most_zeros = zero_ratings['User-ID'].value_counts().idxmax()
user_with_most_zeros

# 2. Делаем предсказание SVD для книг, которым он "поставил" 0
books_with_zero_rating = zero_ratings[zero_ratings['User-ID'] == user_with_most_zeros]['ISBN'].tolist()
svd_model = joblib.load('svd.pkl')
predicted_ratings_svd = []
for book_id in books_with_zero_rating:
    prediction = svd_model.predict(user_with_most_zeros, book_id)
    predicted_ratings_svd.append((book_id, prediction.est))

# 3. Берем те книги, для которых предсказали рейтинг не ниже 8. Считаем, что 8 означет, что книга ему точно понравится.
high_rated_books = [isbn for isbn, rating in predicted_ratings_svd if rating >= 8]

# 4. Делаем предсказание LinReg для этих же книг.
average_ratings = ratings.groupby('ISBN')['Book-Rating'].mean().reset_index()
average_ratings.rename(columns={'Book-Rating': 'Average-Rating'}, inplace=True)

data = pd.merge(books, average_ratings, on='ISBN', how='inner')
recommended_books = data[data['ISBN'].isin(high_rated_books)]

# Признаки и целевая переменная
X = recommended_books[['Book-Author', 'Publisher', 'Year-Of-Publication', 'Book-Title']]
y = recommended_books['Average-Rating']


preprocessor = ColumnTransformer(transformers=[
    ('author_enc', OneHotEncoder(handle_unknown='ignore'), ['Book-Author']),
    ('publisher_enc', OneHotEncoder(handle_unknown='ignore'), ['Publisher']),
    ('title_tfidf', TfidfVectorizer(max_features=500), 'Book-Title'),
    ('scaler', StandardScaler(), ['Year-Of-Publication'])
], remainder='drop')

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# 5. Сортируем полученный на шаге 3 список по убыванию рейтинга линейной модели.
recommended_books['Predicted-Rating'] = model_pipeline.predict(
    recommended_books[['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']]
)

recommended_books_sorted = recommended_books.sort_values(
    by='Predicted-Rating', ascending=False
)[['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication', 'Predicted-Rating']]

print(recommended_books_sorted.head(10))
'''
Рекомендация для пользователя
На основе предсказания, наивыший рейтинг был предсказан моделью для следующих книг:
1. On the Banks of Plum Creek
2. Bloomability
3. Love You Forever
4. The Lion, the Witch and the Wardrobe
5. Hop on Pop
'''