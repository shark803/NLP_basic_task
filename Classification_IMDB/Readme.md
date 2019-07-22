v1 uses the low level tensorflow api to implement the classificator, including the files: config, data_helps,model,train,and eval

v2 and v4 use the estimator api, and share the file config and data_help. The difference is that v2 uses the tf.nn and v4 uses the tf.layer. Attention that the two versions contain the train and test in the model file.


