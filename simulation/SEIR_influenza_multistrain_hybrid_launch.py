import BR_model_test_hybrid as model_test
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import datetime
import random
import json
import os
import csv
import copy
import time
import matplotlib.pyplot as plt
from statistics import mean
from functools import partial
from collections import defaultdict
import multiprocessing as mp
warnings.filterwarnings('ignore')
pd.options.display.width = None
pd.options.display.max_columns = None


# forlorn_v2 - вторичное использование случайных чисел из массива данных
# multistrain - несколько штаммов
# hybrid - рефакторинг для гибридной модели


strains_keys = ['H1N1', 'H3N2', 'B']
infected_init_dic = {'H1N1': 10, 'H3N2': 0, 'B': 0}
POP_MODEL_MOMENT = 500


def strainForInfIndex(idx):  # idx from 1
    return strains_keys[idx-1]


def infIndexForStrain(strain):
    return strains_keys.index(strain) + 1


def aggregateOutputDics(dic_list, strains_keys):
    dic_res = {}
    for key in strains_keys:
        dic_res[key] = []
        for elem in dic_list:
            dic_res[key].append(elem[key])

    return dic_res


def func_b_r(inf_day):
    a = [0.0, 0.0, 0.9, 0.9, 0.55, 0.3, 0.15, 0.05]
    if inf_day < 9:
        return a[inf_day - 1]
    else:
        return 0


# Условие переключения на популяционную модель
def switchToPopModelClause(cur_step, mode="One_traj"):
    if mode == "One_traj":
        return (cur_step == POP_MODEL_MOMENT)

    if mode == "One_traj_sequential":
        return (cur_step > POP_MODEL_MOMENT)

    return (cur_step > POP_MODEL_MOMENT)

# def calcPhiForPop(infected_list): #Подсчитать зараженными разными штаммами
#
#     Phi_arr = np.zeros((len(strains_keys), MAGIC_PREHISTORY_LENGTH))
#     # print("Total people")
#     # print(people_df.shape[0])
#
#     for idx,key in enumerate(strains_keys): #Считаем инфекционных
#         for tau in (1, min(MAGIC_PREHISTORY_LENGTH, POP_MODEL_MOMENT)): #Либо до конца записываемой предыстории, либо до нулевого момента моделирования
#             number_infected_by_strain = infected_list[key][-tau]
#             Phi_arr[idx][tau-1] = number_infected_by_strain
#
#     return Phi_arr


def calcPhiForPop(infected_list):  # Подсчитать зараженными разными штаммами

    len_data = min(MAGIC_PREHISTORY_LENGTH, len(infected_list['H1N1']))
    Phi_arr = np.zeros((len(strains_keys), len_data))
    # print("Total people")
    # print(people_df.shape[0])

    for idx, key in enumerate(strains_keys):  # Считаем инфекционных
        infected_list[key].reverse()
        number_infected_by_strain = infected_list[key]
        number_infected_by_strain_arr = np.array(
            number_infected_by_strain)[:len_data]
        Phi_arr[idx] = number_infected_by_strain_arr.copy()

    return Phi_arr


# Подсчитать полностью иммунных (переболевших в этом сезоне)
def calcRecoveredNumForPop(people_df):
    # print("Recovered")
    dataset_recovered = people_df[(people_df['susceptible_H1N1'] == 0) & (
        people_df['susceptible_H3N2'] == 0) & (people_df['susceptible_B'] == 0)]
    return dataset_recovered.shape[0]


def calcImmuneFractionForPop(people_df):
    ImmuneFraction_list = []
    # print("Total people")
    # print(people_df.shape[0])

    # Считаем тех кто хотя бы к чему-то одному восприимчив, остальные трактуются как переболевшие
    dataset_prior_immune = people_df[
        (people_df['susceptible_H1N1'] == 1) | (people_df['susceptible_H3N2'] == 1) | (people_df['susceptible_B'] == 1)]

    for key in strains_keys:  # Считаем иммунных по каждому штамму отдельно
        # print(key)
        dataset_immune_to_strain = dataset_prior_immune[dataset_prior_immune['susceptible_'+key] == 0]
        # print(dataset_immune_to_strain.shape[0])  #
        ImmuneFraction_list.append(dataset_immune_to_strain.shape[0])

    return ImmuneFraction_list


def main_function(number_seed, dataset, dict_school_id_all, lmbd, days, dict_school_len, res_dir):
    np.random.seed(number_seed)
    data_current = dataset.copy()
    y = {}

    # Штаммы отличаются цифрой в поле infected (от 1 до 3)
    for key in strains_keys:
        # data_susceptible_strain = dataset[dataset["susceptible_" + key] == 1] #Все восприимчивые к данному штамму
        y = np.random.choice(np.array(dataset[dataset["susceptible_"+key] == 1].sp_id),
                             infected_init_dic[key], replace=False)  # Выбираем отсюда же ID больных

        data_current.loc[np.in1d(data_current.sp_id, y), ['infected', 'susceptible_'+str(
            key), 'illness_day']] = [infIndexForStrain(key), 0, 3]  # Раскидываем больных

    # print(data_susceptible[data_susceptible.infected > 0].drop_duplicates())

    id_susceptible_list_dic, latitude_list_dic, longitude_list_dic, type_list_dic, id_place_list_dic, days_inf_dic, results_dic, new_results_dic = {}, {}, {}, {}, {}, {}, {}, {}

    for key in strains_keys:
        id_susceptible_list_dic[key], latitude_list_dic[key], longitude_list_dic[key], type_list_dic[key], id_place_list_dic[
            key], days_inf_dic[key], results_dic[key], new_results_dic[key] = [], [], [], [], [], [], [], []

    # dict_school_id_all.copy() #Первоначальный список домохозяйств с инфицированными
    dict_hh_id = copy.deepcopy(dict_school_id_all)
    dict_hh_id.clear()

    for key in strains_keys:
        dict_hh_id[key] = {i: list(data_current[(data_current.sp_hh_id == i) & (data_current["susceptible_"+key] == 1)].index)  # Восприимчивые в местах, где есть инфицированные
                           for i in data_current.loc[data_current.infected == infIndexForStrain(key), 'sp_hh_id']}  # конкретным штаммом

    # dict_school_id_all.copy()
    dict_work_id = copy.deepcopy(dict_school_id_all)
    dict_work_id.clear()

    for key in strains_keys:
        dict_work_id[key] = {int(i): list(data_current[(data_current.age > 17) & (data_current.work_id == i) & (data_current["susceptible_"+key] == 1)].index)
                             for i in data_current.loc[(data_current.infected == infIndexForStrain(key)) & (data_current.age > 17) & (data_current.work_id != 'X'), 'work_id']}

    [dict_school_id_all[str(i)].remove(j) for i, j in zip(data_current.loc[(data_current.infected > 0) & (data_current.age <= 17) &
                                                                           (data_current.work_id != 'X'), 'work_id'],
                                                          data_current[(data_current.infected > 0) & (data_current.age <= 17) &
                                                                       (data_current.work_id != 'X')].index)]

    # Сначала были все дети в школах, сейчас убрали больных и остались потенциально заразные и иммунные. Нужно убрать иммунных к данному штамму для каждого подсловаря.

    dict_school_id = {}  # Здесь будут списки восприимчивых в местах по штаммам

    # Release version schoolchildren removal
    # for key in strains_keys: #Убираем из школы i детишек j тех, что иммунны к key (оставляем восприимчивых)
    #     dict_school_id[key] = copy.deepcopy(dict_school_id_all)
    #
    #
    #     [dict_school_id[key][str(i)].remove(j) for i, j in
    #      zip(data_current.loc[(data_current["susceptible_"+key] == 0) & (data_current.infected == 0) & (data_current.age <= 17) &
    #                           (data_current.work_id != 'X'), 'work_id'],
    #          data_current[(data_current["susceptible_"+key] == 0) & (data_current.infected == 0) & (data_current.age <= 17) &
    #                       (data_current.work_id != 'X')].index)]

    # Debug version of the upper procedure
    # Убираем из школы i детишек j тех, что иммунны к key (оставляем восприимчивых)
    for key in strains_keys:
        dict_school_id[key] = copy.deepcopy(dict_school_id_all)

        # print("Immune from {}".format(key))
        # print(len(data_current[(data_current["susceptible_" + key] == 0) & (data_current.infected == 0) & (data_current.age <= 17) &
        #                          (data_current.work_id != 'X')]))

        for i, j in zip(
                data_current.loc[(data_current["susceptible_" + key] == 0) & (data_current.infected == 0) & (data_current.age <= 17) &
                                 (data_current.work_id != 'X'), 'work_id'], data_current[(data_current["susceptible_" + key] == 0) & (data_current.infected == 0) &
                                                                                         (data_current.age <= 17) & (data_current.work_id != 'X')].index):
            if not (j in dict_school_id[key][str(i)]):
                print()
                # print("My precious school {}".format(i))
                # print("My child {}".format(j))
            else:
                dict_school_id[key][str(i)].remove(j)

    vfunc_b_r = np.vectorize(func_b_r)

    print("Starting from I0: ")
    print(data_current[data_current.illness_day > 2].sp_id)

    for j in days:  # Главный цикл
        start = time.perf_counter()
        # Есть заразные !!MAGIC NUMBER!! #TODO: убрать
        if len(data_current[data_current.illness_day > 2]) != 0:
            # TODO: генерация с.в. при расходе
            x_rand = np.random.rand(10000000)

            # Заражены каким-то штаммом, нужен для цикла сбора протяженности инфекции
            curr = data_current[data_current.infected > 0]
            print("All inf: ", len(curr))

            hh_inf_dic, work_inf_dic, school_inf_dic = defaultdict(list), defaultdict(list), defaultdict(
                list)

            for _, row in curr.iterrows():  # Сбор id всяких мест с инфицированными
                ill_day = row.illness_day
                if ill_day > 2:
                    # strains_keys[row.infected-1] #0 - H1, 1 - H3, 2 - B   #смотрим, чем болен
                    cur_key = strainForInfIndex(row.infected)
                    # Сбор id домохозяйств с инфицированными
                    hh_inf_dic[row.sp_hh_id, cur_key].append(ill_day)
                    if row.work_id != 'X':
                        if row.age > 17:
                            work_inf_dic[row.work_id, cur_key].append(ill_day)
                        else:
                            school_inf_dic[row.work_id, cur_key].append(
                                ill_day)  # Сбор id школ с инфицированными

            # print("Infected HH")
            # print(hh_inf_dic)
            # print("Infected workplaces")
            # print(work_inf_dic)
            # print("Infected schools")
            # print(school_inf_dic)

            real_inf_hh_dic = {}  # Здесь будем регистрировать вновь заболевших с разбивкой по штаммам
            for key in strains_keys:
                real_inf_hh_dic[key] = np.array([])

            hh_inf_dic_keys_shuffled = list(hh_inf_dic.keys())
            random.shuffle(hh_inf_dic_keys_shuffled)

            # Все домохозяйства, где есть инфицированные любым из штаммов (hh_id, strain)
            for hh_infection_key in hh_inf_dic_keys_shuffled:

                i = hh_infection_key[0]  # id домохозяйства
                cur_strain = hh_infection_key[1]  # штамм

                # Нет данных о жителях в словаре по id домохозяйства для данного штамма (бо добавились инфекции)
                if i not in dict_hh_id[cur_strain].keys():
                    dict_hh_id[cur_strain].update(
                        {i: list(data_current[(data_current.sp_hh_id == i)].index)})
                # Все восприимчивые к конкретному штамму люди в данном домохозяйстве
                hh_len = len(dict_hh_id[cur_strain][i])
                if hh_len != 0:
                    # Инфекционность для всех больных c данным штаммом
                    temp = vfunc_b_r(hh_inf_dic[hh_infection_key])
                    prob = np.repeat(temp, hh_len) * lmbd
                    curr_length = len(prob)
                    hh_rand = x_rand[:curr_length]
                    # Убираем в хвост массива потраченные случайные числа
                    np.roll(x_rand, curr_length)
                    real_inf = len(hh_rand[hh_rand < prob])
                    if hh_len < real_inf:
                        real_inf = hh_len

                    # Выбираем, кто заразился, из восприимчивых
                    real_inf_id = np.random.choice(
                        np.array(dict_hh_id[cur_strain][i]), real_inf, replace=False)
                    # Дописываем инфицированных по штаммам и домохозяйствам
                    real_inf_hh_dic[cur_strain] = np.concatenate(
                        (real_inf_hh_dic[cur_strain], real_inf_id))

                    # print(cur_strain)
                    # print(real_inf_id)
                    # print(real_inf_hh_dic[cur_strain])

                    id_susceptible_list_dic[cur_strain].extend(
                        data_current.sp_id[real_inf_id])  # Выходной txt, списки заболевших
                    type_list_dic[cur_strain].extend(
                        ['household'] * len(real_inf_id))
                    id_place_list_dic[cur_strain].extend(
                        data_current.sp_hh_id[real_inf_id])  # Места, где заболели
                    days_inf_dic[cur_strain].extend([j] * len(real_inf_id))

            real_inf_work_dic = {}

            for key in strains_keys:
                real_inf_work_dic[key] = np.array([])

            some_current = data_current[(data_current.work_id != 'X') & (
                data_current.age > 17)]  # Подможество работающих из всей популяции
            some_current[['work_id']] = some_current[['work_id']].astype(int)

            work_inf_dic_keys_shuffled = list(work_inf_dic.keys())
            random.shuffle(work_inf_dic_keys_shuffled)

            for work_infection_key in work_inf_dic_keys_shuffled:
                i = work_infection_key[0]  # id работы
                cur_strain = hh_infection_key[1]  # штамм

                if i not in dict_work_id[cur_strain].keys():  # Зачем?
                    dict_work_id[cur_strain].update(
                        {i: list(some_current[some_current.work_id == int(i)].index)})

                # Кол-во восприимчивых в месте
                work_len = len(dict_work_id[cur_strain][i])

                if work_len != 0:
                    # Назначаем инфекционность   #Берём инфицированных
                    temp = vfunc_b_r(work_inf_dic[work_infection_key])

                    # Вер. встретить и заразить (lambda*g(tau))
                    prob = np.repeat(temp, work_len) * lmbd
                    curr_length = len(prob)
                    work_rand = x_rand[:curr_length]
                    # Убираем в хвост массива потраченные случайные числа
                    np.roll(x_rand, curr_length)

                    real_inf = len(work_rand[work_rand < prob])
                    if work_len < real_inf:
                        real_inf = work_len
                    real_inf_id = np.random.choice(
                        np.array(dict_work_id[cur_strain][i]), real_inf, replace=False)
                    real_inf_work_dic[cur_strain] = np.concatenate(
                        (real_inf_work_dic[cur_strain], real_inf_id))

                    # print(cur_strain)
                    # print(real_inf_id)
                    # print(real_inf_work_dic[cur_strain])

                    id_susceptible_list_dic[cur_strain].extend(
                        data_current.sp_id[real_inf_id])
                    type_list_dic[cur_strain].extend(
                        ['workplace'] * len(real_inf_id))

                    id_place_list_dic[cur_strain].extend(
                        map(lambda x: int(x), data_current.work_id[real_inf_id]))
                    days_inf_dic[cur_strain].extend([j] * len(real_inf_id))
            # print(number_seed, 'workplaces', len(work_inf), datetime.datetime.now())

            real_inf_school_dic = {}  # Инфекционные в школе и сами такие школы

            for key in strains_keys:
                real_inf_school_dic[key] = np.array([])

            school_inf_dic_keys_shuffled = list(school_inf_dic.keys())
            random.shuffle(school_inf_dic_keys_shuffled)

            for school_infection_key in school_inf_dic_keys_shuffled:
                i = school_infection_key[0]  # id школы
                cur_strain = hh_infection_key[1]  # штамм
                school_len = len(dict_school_id[cur_strain][str(i)])
                if school_len != 0:
                    length = dict_school_len[list(dict_school_id[cur_strain].keys()).index(
                        str(i))]  # Общее число людей
                    temp = vfunc_b_r(school_inf_dic[school_infection_key])
                    # Вероятность контакта фикс индивида с другим
                    prob_cont = 8.5 / (length - 1) if (8.5 + 1) < length else 1
                    res = np.prod(1 - prob_cont * lmbd * temp)
                    real_inf = np.random.binomial(length - 1, 1 - res)
                    if school_len < real_inf:
                        real_inf = school_len

                    real_inf_id = np.random.choice(
                        np.array(dict_school_id[cur_strain][str(i)]), real_inf, replace=False)
                    real_inf_school_dic[cur_strain] = np.concatenate(
                        (real_inf_school_dic[cur_strain], real_inf_id))

                    # print(cur_strain)
                    # print(real_inf_id)
                    # print(real_inf_school_dic[cur_strain])

                    id_susceptible_list_dic[cur_strain].extend(
                        data_current.sp_id[real_inf_id])
                    type_list_dic[cur_strain].extend(
                        ['school'] * len(real_inf_id))

                    id_place_list_dic[cur_strain].extend(
                        map(lambda x: int(x), data_current.work_id[real_inf_id]))  # work_id
                    days_inf_dic[cur_strain].extend([j] * len(real_inf_id))
            # print(number_seed, 'schools', len(school_inf), datetime.datetime.now())

            real_inf_results_dic = {}

            # for key in strains_keys:
            #     real_inf_results_dic[key] = np.array([])

            for key in strains_keys:  # Собираем в одном месте всех, кто заболел
                real_inf_results_dic[key] = np.concatenate(
                    (real_inf_hh_dic[key], real_inf_school_dic[key], real_inf_work_dic[key]))
                real_inf_results_dic[key] = np.unique(
                    real_inf_results_dic[key].astype(int))

            for key in strains_keys:
                # strains_keys.index[cur_strain] + 1 #порядковый номер штамма для графы infected
                data_current.loc[real_inf_results_dic[key], [
                    'infected', 'illness_day', 'susceptible_'+key]] = [infIndexForStrain(key), 1, 0]

            for key in strains_keys:  # Обновление словарей мест с id
                current_hh_id = []
                [current_hh_id.extend(i) for i in dict_hh_id[key].values()]
                check_id = [
                    True if i in current_hh_id else False for i in real_inf_results_dic[key]]
                check_id = [i for i, x in enumerate(check_id) if x is True]
                # Инфицированные домохозяйства
                inf_hh = real_inf_results_dic[key][check_id]
                [dict_hh_id[key][i].remove(j) for i, j in zip(
                    data_current.loc[inf_hh, 'sp_hh_id'], inf_hh)]

                current_wp_id = []
                [current_wp_id.extend(i) for i in dict_work_id[key].values()]
                check_id = [
                    True if i in current_wp_id else False for i in real_inf_results_dic[key]]
                check_id = [i for i, x in enumerate(check_id) if x is True]
                inf_wp = real_inf_results_dic[key][check_id]
                #
                # ##
                # print("Removal pool")
                # print(dict_work_id[key].keys())
                # ##

                # TODO: if j in dict_work_id[key][i] is necessary?
                [dict_work_id[key][i].remove(j) for i, j in zip(
                    data_current.loc[inf_wp, 'work_id'], inf_wp) if j in dict_work_id[key][i]]

                inf_school = real_inf_results_dic[key][(data_current.loc[real_inf_results_dic[key], 'work_id'] != 'X') & (
                    data_current.loc[real_inf_results_dic[key], 'age'] <= 17)]
                [dict_school_id[key][str(i)].remove(j) for i, j in zip(data_current.loc[inf_school, 'work_id'], inf_school)
                 if j in dict_school_id[key][i]]  # TODO: if j in dict_school_id[key][i] is necessary?

        for key in strains_keys:
            newly_infected_by_strain = len(data_current[(data_current.illness_day == 1) & (
                data_current.infected == infIndexForStrain(key))])
            infected_by_strain = len(
                data_current[data_current.infected == infIndexForStrain(key)])

            results_dic[key].append(infected_by_strain)  # Prevalence!!!
            new_results_dic[key].append(newly_infected_by_strain)

            dataset_infected_by_strain = data_current[data_current.infected == infIndexForStrain(key)][[
                'sp_id', 'sp_hh_id']]
            dataset_infected_by_strain = dataset_infected_by_strain.sort_values(by=[
                                                                                'sp_hh_id'])

            # Добавляем координаты к домохозяйствам
            # TWILIGHT ZONE
            # TODO: Debug errors
            # temp = households.loc[households.index.intersection(dataset_infected_by_strain.sp_hh_id), ['latitude', 'longitude']]
            #
            # print("entered")
            # #print(temp.index)
            # #print(dataset_infected_by_strain.index)
            # temp.index = dataset_infected_by_strain.index
            # dataset_infected_by_strain['latitude'] = temp.latitude
            # dataset_infected_by_strain['longitude'] = temp.longitude
            # #print("passed")
            #
            #

            # dataset_infected_by_strain[['sp_id']].to_csv(
            #    r'results/infected_{}_seed_{}_day_{}.txt'.format(number_seed, j, key), sep='\t', index=False)

            # Twilight zone border
            c_inf = int(data_current[data_current.infected ==
                        infIndexForStrain(key)]['infected'].sum())

            print("{}: day {} {} I {}({}) S {} Time {}".format(number_seed, j, key, newly_infected_by_strain, c_inf,
                  int(data_current[['susceptible_'+key]].sum()), datetime.datetime.now()))
        print()
        df_results = pd.DataFrame.from_dict(results_dic)
        df_results.to_csv(res_dir +
                          'temp_results_{}.csv'.format(number_seed), sep='\t', index=False)

        pd.DataFrame(new_results_dic['H1N1']).to_csv(res_dir +
                                                     r'new_infected_{}_seed_{}.txt'.format(number_seed, key), sep='\t', index=False)

        # Switching to compartmental submodel
        if (switchToPopModelClause(j, mode="One_traj")):
            # Prepare init data
            # Берётся из функции Б-Р
            q = [0.0, 0.0, 0.9, 0.9, 0.55, 0.3, 0.15, 0.05]

            # strains = ["A(H1N1)pdm09", "A(H3N2)", "B"]

            Phi_list = calcPhiForPop(new_results_dic)  # Incidence!

            Immune_list = calcImmuneFractionForPop(data_current)
            print("Immune list: ", Immune_list)
            # Считаем, что переболевшие больше не заболеют вообще никаким гриппом
            Recovered_num = calcRecoveredNumForPop(data_current)
            pop_num = data_current.shape[0]
            N = days[-1] - j
            print(N)

            y_model = model_test.launchFromABM(
                Phi_list, Recovered_num, Immune_list, q, lmbd, pop_num, PERCENT_PROTECTED, j+1, N)

            print(y_model)

            for idx, key in enumerate(strains_keys):
                # Нулевое значение пустое, не берём
                results_dic[key].extend(list(y_model[idx])[1:])

            print(results_dic)

            return results_dic

        data_current.loc[data_current.infected >
                         0, 'illness_day'] += 1  # Time goes

        for key in strains_keys:
            # Считаем, что переболевшие иммунны ко всем штаммам
            data_current.loc[data_current.illness_day >
                             8, ['susceptible_'+key]] = 0
        data_current.loc[data_current.illness_day > 8, [
            'infected', 'illness_day']] = 0  # Recovery

        elapsed = (time.perf_counter() - start)

        fname_out_txt = r'out_speed.txt'
        f_handle = open(fname_out_txt, 'a', newline='')
        print("Proc {}, Day {}, time elapsed: {} sec".format(
            mp.current_process().name, j, elapsed))

        writer = csv.writer(f_handle)
        writer.writerow(
            [str(mp.current_process().name), int(j), float(elapsed)])

        # np.savetxt(f_handle, np.column_stack(( str(mp.current_process().name), int(j), float(elapsed) )),
        #            fmt="%s, %d, %f")
        f_handle.close()

    # saving results
    # Twilight zone 2
    # dataset_place_dic = {}
    #
    # for key in strains_keys:
    #     dataset_place_dic[key] = pd.DataFrame({'day': days_inf_dic[key], 'strain': key, 'sp_id': id_susceptible_list_dic[key], 'place_type':  type_list_dic[key], 'place_id': id_place_list_dic[key]})
    #
    #
    #     dataset_place_dic[key].to_csv(r'results/{}_newly_infected_place_{}.txt'.format(number_seed, key),
    #                      sep='\t', index=False)
    # End of the twilight zone

    # [strains_keys[0]], results_dic[strains_keys[1]], results_dic[strains_keys[2]]
    return results_dic, number_seed, lmbd


def gathering_function(sample_size_val, lmbd_val, alpha_val):
    global sample_size
    sample_size = sample_size_val

    global lmbd
    lmbd = lmbd_val

    global alpha
    alpha = alpha_val

    data = pd.read_csv(r'./data/sampled/people_SAMPLED_{}.txt'.format(sample_size),
                       sep='\t', index_col=0)
    data = data[['sp_id', 'sp_hh_id', 'age', 'sex', 'work_id']]
    households = pd.read_csv(
        r'./data/sampled/households_SAMPLED_{}.txt'.format(sample_size), sep='\t')
    households = households[['sp_id', 'latitude', 'longitude']]
    dict_school_id_global = json.load(
        open(os.path.expanduser(r'./data/sampled/schools_SAMPLED_{}.json'.format(sample_size))))

    dict_school_len = [len(dict_school_id_global[i])
                       for i in dict_school_id_global.keys()]

    alpha_dic = {'H1N1': alpha, 'H3N2': 0.74, 'B': 0.6}

    global MAGIC_PREHISTORY_LENGTH
    MAGIC_PREHISTORY_LENGTH = 10

    lmbd = 0.2
    global num_runs
    num_runs = 10
    days = range(1, 100)

    global PERCENT_PROTECTED
    PERCENT_PROTECTED = 1  # Эмуляция разреженной сети, согласно RJNAMM

    res_dir = "results/" + str(sample_size) + "_lmbd=" + \
        str(lmbd) + "_alpha=" + str(alpha_dic['H1N1']) + "/"

    start_all = time.perf_counter()

    try:
        os.makedirs("results/" + str(sample_size) + "_lmbd=" +
                    str(lmbd) + "_alpha=" + str(alpha_dic['H1N1']) + "/")
    except FileExistsError:
        pass

    data[['sp_id', 'sp_hh_id', 'age']] = data[[
        'sp_id', 'sp_hh_id', 'age']].astype(int)
    data[['work_id']] = data[['work_id']].astype(str)
    # data = data.sample(frac=1)  # frac=1
    households[['sp_id']] = households[['sp_id']].astype(int)
    households[['latitude', 'longitude']] = households[[
        'latitude', 'longitude']].astype(float)
    households.index = households.sp_id

    for key in strains_keys:  # По умолчанию все иммунные
        data['susceptible_'+key] = 0

    data['infected'] = 0
    data['illness_day'] = 0

    cpu_num = mp.cpu_count()

    print("{} processors detected".format(cpu_num))

    for key in strains_keys:
        data.loc[np.random.choice(data.index, round(len(
            data) * alpha_dic[key]), replace=False), 'susceptible_'+key] = 1  # Восприимчивые, доля альфа

    output = {}
    with mp.Pool(5) as pool:
        output = pool.map(partial(main_function, dataset=data,
                          dict_school_id_all=dict_school_id_global, lmbd=lmbd, days=days,
                          dict_school_len=dict_school_len, res_dir=res_dir), range(5))

    finish_all = (time.perf_counter() - start_all)
    for i in output:
        print(i[1:])
    print("Total calculation time: {}".format(finish_all))


if __name__ == '__main__':
    sample_size_arr = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
    lmbd_arr = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    alpha_arr = [0.5, 0.6, 0.7, 0.8]
    for sample_size in sample_size_arr:
        for lmbd in lmbd_arr:
            for alpha in alpha_arr:
                gathering_function(sample_size, lmbd, alpha)
