def modify_dict_values(dictionary_list, batch_size):
    """
    주어진 딕셔너리 객체로 이루어진 리스트에서 순차적으로 batch_size 개수만큼 딕셔너리를 선택하여
    해당 딕셔너리의 값을 +1로 변경하는 함수.

    :param dictionary_list: 딕셔너리 객체로 이루어진 리스트
    :param batch_size: 각 반복에서 선택할 딕셔너리의 개수
    :return: 변경된 딕셔너리 객체로 이루어진 리스트
    """
    modified_list = []

    # 리스트에서 batch_size 개수만큼씩 딕셔너리를 선택하여 값을 변경
    for i in range(0, len(dictionary_list), batch_size):
        for j in range(batch_size):
            if i + j < len(dictionary_list):
                modified_dict = dictionary_list[i + j].copy()
                modified_dict['value'] += 1
                modified_list.append(modified_dict)

    return modified_list


# 예시
dict_list = [{'key': 'a', 'value': 1}, {'key': 'b', 'value': 2}, {'key': 'c', 'value': 3}, {'key': 'd', 'value': 4}]
new_dict_list = modify_dict_values(dict_list, 2)
print(dict_list)
print(new_dict_list)
