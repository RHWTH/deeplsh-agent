import numpy as np
import pandas as pd
import math
from scipy import spatial
import pylev

#  将二维的两个栈索引 (a, b)（需满足 a < b）转换为相似度矩阵的一维扁平索引（仅存储上三角区域）。
def get_index_sim(n, a, b) :
    return int(((2*n - (a+1)) * a) / 2 + (b - a) -1)

#  将一维扁平索引 x 转换回二维的两个栈索引 (a, b)（需满足 a < b）。
def get_indices_sim(n, x) :
    
    if x < n - 1 :
        return (0, x + 1)   
    
    else :
        lower_bound = int(n - 1.5 - math.sqrt((n - 1.5)**2 - 2*(x - n + 1)))
        upper_bound = int(n - 0.5 - math.sqrt((n - 0.5)**2 - 2*x)) + 1
        
        tmp = lower_bound
        continu = True
        while continu :
            if ((2*n - (tmp+1)) * tmp) / 2 <= x :
                a = tmp
                tmp += 1
            
            else : 
                continu = False
            
        return (a, int(x - (((2*n - (a+1)) * a) / 2) + a + 1))

#  返回行的索引
def rowIndex(row):
    return row.name

#  根据索引和栈数 n，返回对应的两个栈索引
def get_two_indexes(index, n) :
    return get_indices_sim(n, index)

#  计算两个向量的余弦相似度
def cosine_similarity(vect1, vect2) :
    return 1 - spatial.distance.cosine(vect1, vect2)

#  计算一行与数据框中所有行的余弦相似度
def cosine_similarity_df(df_row, df_copy, index, distinct = True) :
    if distinct :
        return df_copy.apply(lambda x : cosine_similarity(x, df_row), axis = 1).tolist()[index+1:]
    else :
        return df_copy.apply(lambda x : cosine_similarity(x, df_row), axis = 1).tolist()

#  计算两个集合的 Jaccard 相似度
def jaccard (l1, l2) :
    s1 = set(l1)
    s2 = set(l2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


#  计算一行与数据框中所有行的 Jaccard 相似度
def jaccard_df(s, s_copy, index, distinct = True) :
    if distinct :
        return s_copy.apply(lambda x : jaccard(x, s)).tolist()[index+1:]
    else:
        return s_copy.apply(lambda x : jaccard(x, s)).tolist()

#  计算一行与数据框中所有行的编辑距离相似度
def levenshtein_df(s, s_copy, index, distinct = True) :
    if distinct :
        return s_copy.apply(lambda x : 1 - pylev.levenshtein(x, s) / max(len(x), len(s)) ).tolist()[index+1:]
    else :
        return s_copy.apply(lambda x : 1 - pylev.levenshtein(x, s) / max(len(x), len(s)) ).tolist()
   
#  计算一行与数据框中所有行的 Word Mover's Distance 相似度
def wmd_df(s, s_copy, index, model) :
    return s_copy.apply(lambda x : 1 - model.wv.wmdistance(x, s)).tolist()[index+1:]

#  计算两个栈的 PDM 相似度
def pdm(stack1, stack2):
    c = 0.1
    o = 0.1
    stack_len1 = len(stack1)
    stack_len2 = len(stack2)
    if stack_len1 == stack_len2 :
        equal = True
        i = 0
        while equal and i < stack_len1   :
            if stack1[i] == stack2[i] :
                i += 1
            else :
                equal = False
        
    if (stack_len1 == stack_len2) and equal :
        return 1
    else :
        M = [[0. for i in range(stack_len2 + 1)] for j in range(stack_len1 + 1)]

        for i in range(1, stack_len1 + 1):
            for j in range(1, stack_len2 + 1):
                if stack1[i - 1] == stack2[j - 1]:
                    x = math.exp(-c * min(i - 1, j - 1)) * math.exp(-o * abs(i - j))
                else:
                    x = 0.
                M[i][j] = max(M[i - 1][j - 1] + x, M[i - 1][j], M[i][j - 1])
        sig = 0.
        for i in range(min(stack_len1, stack_len2) + 1):
            sig += math.exp(-c * i)
        sim = M[stack_len1][stack_len2] / sig
        return sim
    
#  计算一行与数据框中所有行的 PDM 相似度
def pdm_df(s, s_copy, index, distinct = True) :
    if distinct :
        return s_copy.apply(lambda x : pdm(x, s)).tolist()[index+1:]
    else :
        return s_copy.apply(lambda x : pdm(x, s)).tolist()


#  计算一行与数据框中所有行的 Brodie 相似度
def brodie_df(s, s_copy, index, df_bag_of_frames, distinct = True):
    if distinct :
        return s_copy.apply(lambda x : brodie_similarity(s, x, df_bag_of_frames, match = 1, mismatch = 1, gap = 0)).tolist()[index+1:]
    else :
        return s_copy.apply(lambda x : brodie_similarity(s, x, df_bag_of_frames, match = 1, mismatch = 1, gap = 0)).tolist()

# 计算两个栈序列的 Brodie 相似度        
def brodie_similarity (x, y, df_bag_of_frames, match, mismatch, gap):

    x = [process_frame(frame) for frame in x]
    y = [process_frame(frame) for frame in y]
    nx = len(x)
    ny = len(y)

    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    F[0,:] = np.linspace(0, -ny * gap, ny + 1)

    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                p1 = 1 - df_bag_of_frames[x[i]].sum() / df_bag_of_frames.shape[0]
                p2 = 1 - i / nx
                p3 = math.exp(-abs(i - j) / 2)
                p = p1 * p2 * p3
                t[0] = F[i,j] + p
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
    
    ratio_factor_stack1 = 0
    for i in range(nx):
        if x[i] in df_bag_of_frames.columns :
            ratio_factor_stack1 += (1 - df_bag_of_frames[x[i]].sum() / df_bag_of_frames.shape[0]) * (1 - i / nx)
        else :
            ratio_factor_stack1 += (1 - i / nx)

    ratio_factor_stack2 = 0
    for j in range(ny):
        if y[j] in df_bag_of_frames.columns :
            ratio_factor_stack2 += (1 - df_bag_of_frames[y[j]].sum() / df_bag_of_frames.shape[0]) * (1 - j / nx)
        else :
            ratio_factor_stack2 += (1 - j / nx)
    
    return F[-1][-1] / max(ratio_factor_stack1, ratio_factor_stack2)

# 处理帧名称
def process_frame(frame) :
    frame = frame.lower()
    frame = frame.replace('$','')
    frame = frame.replace('/','')
    frame = frame.replace('<','')
    frame = frame.replace('>','')
    return frame
    
# 基于 Needleman-Wunsch 全局序列比对算法，计算两个栈轨迹的相似度
def nw_similarity (df_distinct_stacks, idStack1, idStack2, match = 1, mismatch = 1, gap = 1):
    x = df_distinct_stacks['listStackTrace'][idStack1]
    y = df_distinct_stacks['listStackTrace'][idStack2]
    x = [process_frame(frame) for frame in x]
    y = [process_frame(frame) for frame in y]
    nx = len(x)
    ny = len(y)

    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    F[0,:] = np.linspace(0, -ny * gap, ny + 1)

    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
    
    return F[-1][-1] / max(nx, ny)

# 计算两个栈序列的 Lerch 相似度 ——基于 TF-IDF 的栈帧匹配
def lerch(stack1, stack2, dict_idf_frames):
    stack1 = [process_frame(frame) for frame in stack1]
    stack2 = [process_frame(frame) for frame in stack2]
    score = 0
    max_score = 0
    for frame in stack1:
        tf = math.sqrt(stack2.count(frame))            
        idf = dict_idf_frames[frame]
        score += tf * idf**2
        if tf > 0 :
            max_score += tf * idf**2
        else :
            max_score += math.sqrt(stack1.count(frame)) * idf**2
    return score / max_score

# 批量计算栈序列的 Lerch 相似度
def lerch_df(df_row, df_copy, index, dict_idf_frames, distinct = True) :
    if distinct :
        return df_copy.apply(lambda x : lerch(x, df_row, dict_idf_frames)).tolist()[index+1:]
    else :
        return df_copy.apply(lambda x : lerch(x, df_row, dict_idf_frames)).tolist()

# 计算两个栈序列的 Moroo 相似度 ——基于 Lerch 和 PDM 的加权融合
def moroo(stack1, stack2, dict_idf_frames, alpha):
    stack1 = [process_frame(frame) for frame in stack1]
    stack2 = [process_frame(frame) for frame in stack2]
    score_lerch = lerch(stack1, stack2, dict_idf_frames)
    score_rebucket = pdm(stack1, stack2)
    
    if score_lerch == 0 and score_rebucket == 0 :
        return 0
    else :
        return (score_lerch * score_rebucket) / (alpha * score_rebucket + (1 - alpha) * score_lerch)

# 计算一行与数据框中所有行的 Moroo 相似度
def moroo_df(df_row, df_copy, index, dict_idf_frames, alpha, distinct = True) :
    if distinct :
        return df_copy.apply(lambda x : moroo(x, df_row, dict_idf_frames, alpha)).tolist()[index+1:]
    else :
        return df_copy.apply(lambda x : moroo(x, df_row, dict_idf_frames, alpha)).tolist()
    
# 计算两个栈序列的前缀匹配相似度
def prefix_match(stack1 , stack2):
    prefix_len = 0
    for i, (frame1, frame2) in enumerate(zip(stack1, stack2)):
        if frame1 != frame2 :
            prefix_len = i
            break
    return prefix_len / max(len(stack1), len(stack2))

# 计算一行与数据框中所有行的前缀匹配相似度
def prefix_match_df(df_row, df_copy, index, distinct = True) :
    if distinct :
        return df_copy.apply(lambda x : prefix_match(x, df_row)).tolist()[index+1:]
    else :
        return df_copy.apply(lambda x : prefix_match(x, df_row)).tolist()
        

# 计算栈序列的加权
def get_weight_stack(stack, dict_idf_frames, alpha, beta, gamma) :

    local_weights = [1 / (1 + i) ** alpha for i, _ in enumerate(stack)]
    global_weights = []
    for frame in stack :
        if frame in dict_idf_frames :
            global_weights.append(1 / (1 + math.exp(-beta * (dict_idf_frames[frame] - gamma))))
        else :
            global_weights.append(1 / (1 + math.exp(beta * gamma)))
    return [lw * gw for lw, gw in zip(local_weights, global_weights)]  

# 计算加权编辑距离
def levenshtein_dist_weights(frames1, weights1, frames2, weights2) :
    matrix = [[0.0 for _ in range(len(frames1) + 1)] for _ in range(len(frames2) + 1)]

    prev_column = matrix[0]

    for i in range(len(frames1)):
        prev_column[i + 1] = prev_column[i] + weights1[i]

    if len(frames1) == 0 or len(frames2) == 0:
        return 0.0

    curr_column = matrix[1]

    for i2 in range(len(frames2)):

        frame2 = frames2[i2]
        weight2 = weights2[i2]

        curr_column[0] = prev_column[0] + weight2

        for i1 in range(len(frames1)):

            frame1 = frames1[i1]
            weight1 = weights1[i1]

            if frame1 == frame2:
                curr_column[i1 + 1] = prev_column[i1]
            else:
                change = weight1 + weight2 + prev_column[i1]
                remove = weight2 + prev_column[i1 + 1]
                insert = weight1 + curr_column[i1]

                curr_column[i1 + 1] = min(change, remove, insert)

        if i2 != len(frames2) - 1:
            prev_column = curr_column
            curr_column = matrix[i2 + 2]

    return curr_column[-1]

# 计算加权编辑距离相似度
def traceSim(stack1, stack2, dict_idf_frames, alpha, beta, gamma):
    
    stack1 = [process_frame(frame) for frame in stack1]
    stack2 = [process_frame(frame) for frame in stack2]
    
    stack1_weights = get_weight_stack(stack1, dict_idf_frames, alpha, beta, gamma)
    stack2_weights = get_weight_stack(stack2, dict_idf_frames, alpha, beta, gamma)
    
    max_dist = sum(stack1_weights) + sum(stack2_weights)
    dist = levenshtein_dist_weights(stack1, stack1_weights, stack2, stack2_weights)
    sim = 0 if max_dist == 0 else 1 - dist / max_dist
    return sim


# 计算一行与数据框中所有行的加权编辑距离相似度
def traceSim_df(df_row, df_copy, index, dict_idf_frames, alpha = 0.5, beta = 1, gamma = 0, distinct = True) :
    if distinct :
        return df_copy.apply(lambda x : traceSim(x, df_row, dict_idf_frames, alpha, beta, gamma)).tolist()[index+1:]
    else :
        return df_copy.apply(lambda x : traceSim(x, df_row, dict_idf_frames, alpha, beta, gamma)).tolist()