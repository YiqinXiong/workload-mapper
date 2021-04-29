# -*- coding: utf-8 -*-
# Author: 熊逸钦
# Time: 2021/4/29 16:04

import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import joblib

workspace_dir = f"/home/{os.getlogin()}/workspace/"
logs_dir = f"{workspace_dir}HUST-RocksDB-AITuner-GraduationDesign/auto-tester/ycsb_logs/"


def get_data_set():
    X = None
    y = []
    cmd = f"find {logs_dir}*-ANOVA/ -name \"*.log\""
    log_path_list = os.popen(cmd).readlines()
    for log_path in log_path_list:
        wl = log_path.split('/')[9]
        if wl in ['longscan', 'pntlookup80', 'shortscan', 'workloada', 'writeheavy']:
            xx = get_characteristic_from_log_file(log_path[:-1])
            # print(X, xx)
            if X is not None:
                X = np.vstack((X, xx))
            else:
                X = np.array(xx)
            y.append(wl)
    return X, y


def get_characteristic_from_log_file(log_path):
    # 部分性能指标
    read_keys = 0
    write_keys = 0
    scan_seek = 0
    scan_next = 0
    # 读取log内容
    with open(log_path, "r") as f:
        logs = f.readlines()
        logs.reverse()
        for line in logs:
            if line.startswith('Cumulative scans:'):
                sl = line.split()
                scan_seek = float(sl[2])
                scan_next = float(sl[6])
            elif line.startswith('Cumulative reads:'):
                sl = line.split()
                read_keys = float(sl[2])
            elif line.startswith('Cumulative writes:'):
                sl = line.split()
                write_keys = float(sl[4])
                break
    # 统计数据
    sum_count = float(read_keys) + float(write_keys) + scan_next
    read_rate = float(read_keys) / sum_count
    write_rate = float(write_keys) / sum_count
    scan_rate = scan_next / sum_count
    scan_length = scan_next / max(scan_seek, 1)
    xx = np.array([read_rate, write_rate, scan_rate, scan_length])
    return xx


if __name__ == '__main__':
    data, label = get_data_set()
    # y = range(5)
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.3, random_state=42)
    clf = GaussianNB()
    # 拟合数据
    clf.fit(train_X, train_y)
    joblib.dump(clf, "./workload_mapper.pkl")
    clf2 = joblib.load("./workload_mapper.pkl")
    score = clf2.score(test_X, test_y)
    print(score)
