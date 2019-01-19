from utils import *

if __name__ == '__main__':
    # Create TF Records
    train, val, test = get_db_sets('db')
    create_tf_record(train, 'train')
    create_tf_record(val, 'val')
    create_tf_record(test, 'test')
