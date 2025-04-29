import os.path


import numpy as np
import pandas as pd
from helper import sliding_window, fixedSize_window, structure_log

#### for Thunderbird, Liberty, BGL


data_dir = r'/mnt/public/gw/SyslogData/BGL'
log_name = "BGL.log"

start_line = 0
end_line = None

# # Liberty
# start_line = 40000000
# end_line = 45000000

# # thunderbird
# start_line = 160000000
# end_line = 170000000

output_dir = data_dir



if __name__ == '__main__':
    # group_type = 'time_sliding'

    window_size = 100
    step_size = 100

    # if 'bgl' in log_name.lower():
    #     window_size = 5  # 5 minutes
    #     step_size = 1  # 1  minutes
    # elif 'thunderbird' in log_name.lower():
    #     window_size = 1 # 1 minutes
    #     step_size = 0.5 # 0.5  minutes
    # else:
    #     raise Exception('missing valid window_size and step_size')

    if 'thunderbird' in log_name.lower() or 'spirit' in log_name.lower() or 'liberty' in log_name.lower():
        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'   #thunderbird  , spirit, liberty
    elif 'bgl' in log_name.lower():
        log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'  #bgl
    else:
        raise Exception('missing valid log format')
    print(f'Auto log_format: {log_format}')

    structure_log(data_dir, output_dir, log_name, log_format, start_line = start_line, end_line = end_line)

    print(f'window_size: {window_size}; step_size: {step_size}')

    train_ratio = 0.8

    df = pd.read_csv(os.path.join(output_dir,f'{log_name}_structured.csv'))

    print(len(df))

    # data preprocess
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

    # if group_type == 'time_sliding':
    #     df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    #     df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
    #     df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    #     df['deltaT'].fillna(0)

    train_len = int(train_ratio*len(df))

    df_train = df[:train_len]

    df_test = df[train_len:]
    df_test = df_test.reset_index(drop=True)

    print('Start grouping.')

    # grouping with fixedSize window
    session_train_df = fixedSize_window(
        df_train[['Content', 'Label']],
        window_size=window_size, step_size=step_size
    )

    # grouping with fixedSize window
    session_test_df = fixedSize_window(
        df_test[['Content', 'Label']],
        window_size=window_size, step_size=step_size
    )

    # if group_type == 'time_sliding':
    #     # grouping with time sliding window
    #     session_train_df = sliding_window(df_train[["timestamp", "Label", "deltaT",'Content']],
    #                                 para={"window_size": int(window_size)*60, "step_size": int(step_size) * 60}
    #                                 )
    #
    #     # grouping with time sliding window
    #     session_test_df = sliding_window(df_test[["timestamp", "Label", "deltaT",'Content',]],
    #                                 para={"window_size": int(window_size)*60, "step_size": int(step_size) * 60}
    #                                 )
    # else:
    #     # grouping with fixedSize window
    #     session_train_df = fixedSize_window(
    #         df_train[['Content', 'Label']],
    #         window_size = window_size, step_size = step_size
    #         )
    #
    #     # grouping with fixedSize window
    #     session_test_df = fixedSize_window(
    #         df_test[['Content', 'Label']],
    #         window_size = window_size, step_size = step_size
    #         )

    col = ['Content', 'Label','item_Label']
    spliter=' ;-; '

    session_train_df = session_train_df[col]
    session_train_df['session_length'] = session_train_df["Content"].apply(len)
    session_train_df["Content"] = session_train_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_train_len = session_train_df['session_length'].mean()
    max_session_train_len = session_train_df['session_length'].max()
    num_anomalous_train= session_train_df['Label'].sum()
    num_normal_train = len(session_train_df['Label']) - session_train_df['Label'].sum()

    session_test_df = session_test_df[col]
    session_test_df['session_length'] = session_test_df["Content"].apply(len)
    session_test_df["Content"] = session_test_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_test_len = session_test_df['session_length'].mean()
    max_session_test_len = session_test_df['session_length'].max()
    num_anomalous_test= session_test_df['Label'].sum()
    num_normal_test = len(session_test_df['Label']) - session_test_df['Label'].sum()


    session_train_df.to_csv(os.path.join(output_dir, 'train.csv'),index=False)
    session_test_df.to_csv(os.path.join(output_dir, 'test.csv'),index=False)

    print('Train dataset info:')
    print(f"max session length: {max_session_train_len}; mean session length: {mean_session_train_len}\n")
    print(f"number of anomalous sessions: {num_anomalous_train}; number of normal sessions: {num_normal_train}; number of total sessions: {len(session_train_df['Label'])}\n")

    print('Test dataset info:')
    print(f"max session length: {max_session_test_len}; mean session length: {mean_session_test_len}\n")
    print(f"number of anomalous sessions: {num_anomalous_test}; number of normal sessions: {num_normal_test}; number of total sessions: {len(session_test_df['Label'])}\n")

    with open(os.path.join(output_dir, 'train_info.txt'), 'w') as file:
        # 写入内容到文件
        file.write(f"max session length: {max_session_train_len}; mean session length: {mean_session_train_len}\n")
        file.write(f"number of anomalous sessions: {num_anomalous_train}; number of normal sessions: {num_normal_train}; number of total sessions: {len(session_train_df['Label'])}\n")

    with open(os.path.join(output_dir, 'test_info.txt'), 'w') as file:
        # 写入内容到文件
        file.write(f"max session length: {max_session_test_len}; mean session length: {mean_session_test_len}\n")
        file.write(f"number of anomalous sessions: {num_anomalous_test}; number of normal sessions: {num_normal_test}; number of total sessions: {len(session_test_df['Label'])}\n")

