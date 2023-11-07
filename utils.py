def write_log(log_file, s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s + '\n')
    