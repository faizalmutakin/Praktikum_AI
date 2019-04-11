kfold=StratifiedKFold(n_splits=5)
splits=kfold.split(d, d['CLASS'])

model = Sequential()
model.add(Dense(512, input_shape=(2000 ,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adamax', metrics=['accuracy'])

d_train_inputs = d_train_inputs/np.amax(np.absolute(d_train_inputs))
d_test_inpus = d_test_inpus/np.amax(np.absolute(d_test_inputs))

d_train_inputs = tokenizer.texts_to_matrix(train_content, mode='tfidf')
d_test_inpus = tokenizer.texts_to_matrix(test_content, mode='tfidf')