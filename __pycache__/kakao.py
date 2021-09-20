
ans_list = []

def recur(info, lion_info, idx, total_arrow, left_arrow):

    global ans_list
    
    if idx == 9:
        if sum(lion_info) == total_arrow:
            print(lion_info)
            ans_list.append(lion_info)
        ans_list
    
    if left_arrow >= info[idx] + 1:
        lion_info[idx] = info[idx] + 1
        recur(info, lion_info, idx + 1, total_arrow, left_arrow - info[idx] - 1)
        lion_info[idx] = 0
    
    recur(info, lion_info, idx + 1, total_arrow, left_arrow)


def solution(n, info):
    answer = []
    # n = 5
    # info = [2,1,1,1,0,0,0,0,0,0,0]
    
    global ans_list
    
    lion_info = [0,0,0,0,0,0,0,0,0,0,0]
    print(recur(info, lion_info, 0, n, n))
    
    print(ans_list)
    
    return answer

n = 5
info = [2,1,1,1,0,0,0,0,0,0,0]
solution(n, info)