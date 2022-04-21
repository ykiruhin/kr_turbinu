from iapws import IAPWS97 as WSP
from streamlit_option_menu import option_menu
import math as M
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from sympy import *

st.set_page_config(page_title="Курсовая работа", page_icon=":sunglasses:", layout = "wide")
with st.container():
    st.subheader('Кирюхин Ярослав, Фпэ-01-19, Вариант 5. :wave:')

selected = option_menu(
        menu_title="Меню",
        options=["Курсовая работа","Задание 1", "Задание 2","Задание 3"],
        icons=["award-fill","book","book","book"],
        menu_icon="home",
        default_index=0,
        orientation ="horizontal"
    )
if selected == "Курсовая работа":
    st.subheader("Спроектировать эскиз проточной части ЦВД паровой турбины."
                 "Исходные данные представляются в виде таблицы и выбираются по номеру журнала."
                 "")
    st.subheader("При выполнении КР провести следующие работы:")
    st.markdown("1. Построить процесс расширения пара в турбине. "
                "Определение расходов пара на входе в турбину (G0) и в конденсатор (Gк). "
                "Получить зависимость КПД ПТУ от параметра заданного в таблице.")
    st.markdown("2. Проведение расчета регулирующей ступени и определение зависимости ηол от U/cф")
    st.markdown("3. Определение числа ступеней и распределение параметров по ним.")
    # чтение из файла:
    Dano = pd.read_excel('file.xlsx')
    option = st.selectbox(
        'Выберете ФИО сстудента',
        Dano["ФИО"])
    mas = []
    st.write('Выбран:', option)
    # Dano.loc[Dano['ФИО'] == option]
    mas = Dano.loc[Dano['ФИО'] == option]
    itog = pd.DataFrame({
        "P0, МПа": mas['P0, МПа'],
        "t0, C": mas['t0, C']
    })
    itog
if selected == "Задание 1":
    with st.container():
        left_column, right_column = st.columns([1, 1.2])
        with left_column:
            st.subheader('Исходные Данные:')
            st.markdown('Турбина: Т-180/215-12.8-2 ЛМЗ')
            st.markdown('Ne = 216 МВт')
            st.markdown("p0 = 12.9 МПа")
            st.markdown('t0 = 546 C')
            st.markdown('ppp = 2.48 МПа')
            st.markdown('tpp = 556 C')
            st.markdown('pk = 2-10 кПа')
            st.markdown('tpv = 238 C')
            st.markdown('z = 8 шт.')
        with right_column:
            st.subheader('Корректировка данных:')
            Ne = st.number_input('Ne, МВт ', value=216) * 10 ** 6
            p0 = st.number_input('p0, МПа ', value = 12.9) * 10 ** 6
            t0 = st.number_input('t0, C ', value=546)
            T0 = t0 + 273.15  # K
            ppp = st.number_input('ppp, МПа ', value=2.48) * 10 ** 6
            tpp = st.number_input('tpp, C ', value=556)
            Tpp = tpp + 273.15  # K
            tpv = st.number_input('tpv, C ', value=238)
            Tpv = tpv + 273.15  # K

    sld = st.slider('Давление в конденсаторе, Па:', min_value=2000, max_value=10000, value=10000, step=500)
    sld = sld + 0.001
    pk = list(np.arange(2000, sld, 500))

    pk_min = 2e3  # кПа

    delta_p_0 = 0.05 * p0
    delta_p_pp = 0.1 * ppp
    delta_p = 0.03 * ppp
    from bokeh.plotting import figure

    def Calculate_G0_Gk(N_e, p_0, T_0, p_pp, T_pp, p_k, T_pv):
        # Потери:
        d_p0 = 0.05
        d_p_pp = 0.1
        d_p = 0.03
        # Параметры свежего пара
        point_0 = WSP(P=p_0 * 10 ** (-6), T=T_0)
        s_0 = point_0.s
        h_0 = point_0.h
        v_0 = point_0.v
        #
        p_0_ = p_0 - 0.05 * p_0
        point_p_0_ = WSP(P=p_0_ * 10 ** (-6), h=h_0)
        t_0_ = point_p_0_.T - 273.15
        s_0_ = point_p_0_.s
        v_0_ = point_p_0_.v
        # Теоретический процесс расширения в ЦВД
        p_1t = p_pp + 0.1 * p_pp
        point_1t = WSP(P=p_1t * 10 ** (-6), s=s_0)
        t_1t = point_1t.T - 273.15
        h_1t = point_1t.h
        v_1t = point_1t.v
        #
        point_pp = WSP(P=p_pp * 10 ** (-6), T=T_pp)
        h_pp = point_pp.h
        s_pp = point_pp.s
        v_pp = point_pp.v
        # Действительный процесс расширения в ЦВД
        H_0 = h_0 - h_1t
        eta_oi = 0.85
        H_i_cvd = H_0 * eta_oi
        h_1 = h_0 - H_i_cvd
        point_1 = WSP(P=p_1t * 10 ** (-6), h=h_1)
        s_1 = point_1.s
        T_1 = point_1.T
        v_1 = point_1.v
        #
        p_pp_ = p_pp - 0.03 * p_pp
        point_pp_ = WSP(P=p_pp_ * 10 ** (-6), h=h_pp)
        s_pp_ = point_pp_.s
        v_pp_ = point_pp_.v
        #
        point_kt = WSP(P=p_k * 10 ** (-6), s=s_pp)
        T_kt = point_kt.T
        h_kt = point_kt.h
        v_kt = point_kt.v
        s_kt = s_pp
        #
        H_0_csdcnd = h_pp - h_kt
        eta_oi = 0.85
        H_i_csdcnd = H_0_csdcnd * eta_oi
        h_k = h_pp - H_i_csdcnd
        point_k = WSP(P=p_k * 10 ** (-6), h=h_k)
        T_k = point_k.T
        s_k = point_k.s
        v_k = point_k.v
        #
        point_k_v = WSP(P=p_k * 10 ** (-6), x=0)
        h_k_v = point_k_v.h
        s_k_v = point_k_v.s
        eta_oiI = (h_1 - h_0) / (h_1t - h_0)
        p_pv = 1.4 * p_0
        point_pv = WSP(P=p_pv * 10 ** (-6), T=T_pv)
        h_pv = point_pv.h
        s_pv = point_pv.s
        #
        ksi_pp_oo = 1 - (1 - (T_k * (s_pp - s_k_v)) / ((h_0 - h_1t) + (h_pp - h_k_v))) / (
                    1 - (T_k * (s_pp - s_pv)) / ((h_0 - h_1t) + (h_pp - h_pv)))
        T_0_ = WSP(P=p_pv * 10 ** (-6), x=0).T
        T_ = (point_pv.T - point_k.T) / (T_0_ - point_k.T)
        if T_ <= 0.636363636:
            ksi1 = -1.53 * T_ ** 2 + 2.1894 * T_ + 0.0048
        elif 0.636363636 < T_ <= 0.736363636:
            ksi1 = -1.3855 * T_ ** 2 + 2.0774 * T_ + 0.0321
        elif 0.736363636 < T_ <= 0.863636364:
            ksi1 = -19.459 * T_ ** 3 + 44.048 * T_ ** 2 - 33.037 * T_ + 9.0511
        else:
            ksi1 = 21321
        if T_ <= 0.631818182:
            ksi2 = -1.7131 * T_ ** 2 + 2.3617 * T_ - 0.0142
        elif 0.631818182 < T_ <= 0.718181818:
            ksi2 = -2.5821 * T_ ** 2 + 3.689 * T_ - 0.4825
        elif 0.718181818 < T_ <= 0.827272727:
            ksi2 = -1.9864 * T_ ** 2 + 3.138 * T_ - 0.3626
        else:
            ksi2 = 21321
        ksi = (ksi1 + ksi2) / 2
        ksi_r_pp = ksi * ksi_pp_oo
        eta_ir = (H_i_cvd + H_i_csdcnd) / (H_i_cvd + (h_pp - h_k_v)) * 1 / (1 - ksi_r_pp)
        H_i = eta_ir * ((h_0 - h_pv) + (h_pp - h_1))
        eta_m = 0.994
        eta_eg = 0.99
        G_0 = N_e / (H_i * eta_m * eta_eg * (10 ** 3))
        G_k = N_e / ((h_k - h_k_v) * eta_m * eta_eg * (10 ** 3)) * (1 / eta_ir - 1)
        return eta_ir, G_0, G_k

    a1, a2, a3 = [], [], []
    for p in pk:
        a = Calculate_G0_Gk(N_e=Ne, p_0=p0, T_0=T0, p_pp=ppp, T_pp=Tpp, p_k=p, T_pv=Tpv)
        a1.append(a[0])
        a2.append(a[1])
        a3.append(a[2])
    with st.container():
        left_column, right_column = st.columns([1, 1.3])
        with left_column:
            itog = pd.DataFrame({
                "Давление в конденсаторе, Па": (list(np.arange(2000, sld, 500))),
                "КПД": a1,
                "G_0, кг/c": a2,
                "G_k, кг/c": a3,
            })
            itog

            st.write("Максимальный КПД:")
            itog.iloc[0:1]

        with right_column:
            x = (list(np.arange(2000, sld, 500)))
            y = (a1)

            p = figure(
                title='Зависимость КПД от давления в конденсаторе',
                x_axis_label='давление в конденсаторе, Па',
                y_axis_label='КПД')

            p.line(x, y, legend_label='Зависимость КПД от давления в конденсаторе', line_width=3)
            st.bokeh_chart(p, use_container_width=True)

    with st.container():
        left_column, right_column = st.columns([1.2, 1])
        with left_column:
            fighs = plt.figure()
            point_0 = WSP(P=p0 * 1e-6, T=T0)
            p_0_d = p0 - delta_p_0
            point_0_d = WSP(P=p_0_d * 1e-6, h=point_0.h)
            p_1t = ppp + delta_p_pp
            point_1t = WSP(P=p_1t * 10 ** (-6), s=point_0.s)
            H_01 = point_0.h - point_1t.h
            kpd_oi = 0.85
            H_i_cvd = H_01 * kpd_oi
            h_1 = point_0.h - H_i_cvd
            point_1 = WSP(P=p_1t * 1e-6, h=h_1)
            point_pp = WSP(P=ppp * 1e-6, T=Tpp)
            p_pp_d = ppp - delta_p_pp
            point_pp_d = WSP(P=p_pp_d * 1e-6, h=point_pp.h)
            point_kt = WSP(P=pk_min * 1e-6, s=point_pp.s)
            H_02 = point_pp.h - point_kt.h
            kpd_oi = 0.85
            H_i_csd_cnd = H_02 * kpd_oi
            h_k = point_pp.h - H_i_csd_cnd
            point_k = WSP(P=pk_min * 1e-6, h=h_k)

            s_0 = [point_0.s - 0.05, point_0.s, point_0.s + 0.05]
            h_0 = [WSP(P=p0 * 1e-6, s=s_).h for s_ in s_0]
            s_1 = [point_0.s - 0.05, point_0.s, point_0.s + 0.18]
            h_1 = [WSP(P=p_1t * 1e-6, s=s_).h for s_ in s_1]
            s_0_d = [point_0_d.s - 0.05, point_0_d.s, point_0_d.s + 0.05]
            h_0_d = h_0
            s_pp = [point_pp.s - 0.05, point_pp.s, point_pp.s + 0.05]
            h_pp = [WSP(P=ppp * 1e-6, s=s_).h for s_ in s_pp]
            s_k = [point_pp.s - 0.05, point_pp.s, point_pp.s + 0.8]
            h_k = [WSP(P=pk_min * 1e-6, s=s_).h for s_ in s_k]
            s_pp_d = [point_pp_d.s - 0.05, point_pp_d.s, point_pp_d.s + 0.05]
            h_pp_d = h_pp

            plt.plot([point_0.s, point_0.s, point_0_d.s, point_1.s], [point_1t.h, point_0.h, point_0.h, point_1.h],
                     '-or')
            plt.plot([point_pp.s, point_pp.s, point_pp_d.s, point_k.s], [point_kt.h, point_pp.h, point_pp.h, point_k.h],
                     '-or')
            plt.plot(s_0, h_0)
            plt.plot(s_1, h_1)
            plt.plot(s_0_d, h_0_d)
            plt.plot(s_pp, h_pp)
            plt.plot(s_k, h_k)
            plt.plot(s_pp_d, h_pp_d)

            for x, y, ind in zip([point_pp.s, point_k.s], [point_pp.h, point_k.h], ['{пп}', '{к}']):
                plt.text(x - 0.45, y + 40, '$h_' + ind + ' = %.3f $' % y)
            for x, y, ind in zip([point_kt.s, point_pp_d.s], [point_kt.h, point_pp_d.h], ['{кт}', '{ппд}']):
                plt.text(x + 0.03, y + 40, '$h_' + ind + ' = %.3f $' % y)

            for x, y, ind in zip([point_0.s, point_1.s], [point_0.h, point_1.h], ['{0}', '{1}']):
                plt.text(x - 0.01, y + 120, '$h_' + ind + ' = %.3f $' % y)

            for x, y, ind in zip([point_1t.s, point_0_d.s], [point_1t.h, point_0_d.h], ['{1т}', '{0д}']):
                plt.text(x + 0.03, y - 60, '$h_' + ind + ' = %.3f $' % y)

            plt.title("h - s диаграмма")
            plt.xlabel("s, кДж/(кг*K)")
            plt.ylabel("h, кДж/кг")
            plt.grid(True)
            st.pyplot(fighs)
        with right_column:
            st.subheader('* h-s диаграмма составлена для p = 2 кПа')
if selected == "Задание 2":
    with st.container():
        left_column, right_column = st.columns([1,1.2])
        with left_column:
            st.subheader('Исходные Данные:')
            st.markdown('Турбина: Т-180/215-12.8-2 ЛМЗ')
            st.markdown('Давление пара перед ступенью: p0 = 12.9 МПа')
            st.markdown('Температура пара перед ступенью: t0 = 546 C')
            st.markdown('Расход пара: G_0 = 243.13 кг/c')
            st.markdown('Средний диаметр: d = 1.04 м')
            st.markdown('Частота вращения: n = 60 Гц')
            st.markdown('Степень реактивности rho = 0.05')
            st.markdown('Теплоперепад: H_0 = 90 - 110 кДж/кг')
            st.markdown('хорда сопловой лопатки: b_1 = 0.06 м')
            st.markdown('хорда рабочей лопатки: b_2 = 0.03 м')
            st.markdown('Высота лопатки: l_1 = 0.015 м')
            st.markdown('Угол направления скорости c1t: alpha_1 = 12 градусов')
            st.markdown('Перекрышка: delta = 0.003 градусов')
            st.markdown('kappa_vs = 0')
        with right_column:
            st.subheader('Корректировка данных:')
            p_0 = st.number_input('p0, МПа ', value = 12.9)
            t0 = st.number_input('t0, C ', value = 546)
            T_0 = t0 + 273.15  # K
            G_0 = st.number_input('G_0, кг/с ', value=243.13)
            d = st.number_input('d, м ', value=1.04)
            n = st.number_input('n, Гц ', value=60)
            rho = st.number_input('rho ', value=0.05)
            b_1 = st.number_input('b_1, м ', value=0.06)
            b_2 = st.number_input('b_2, м ', value=0.03)
            l_1 = st.number_input('l_1, м ', value=0.015)
            alpha_1 = st.number_input('alpha_1, градусов ', value=12)
            delta = 0.003
            kappa_vs =0

    H_0 = 90 #кДж/кг

    def iso_bar(wsp_point, min_s=-0.1, max_s=0.11, step_s=0.011, color = 'r'):
        if not isinstance(wsp_point,list):
            iso_bar_0_s = np.arange(wsp_point.s+min_s,wsp_point.s+max_s,step_s).tolist()
            iso_bar_0_h = [WSP(P = wsp_point.P, s = i).h for i in iso_bar_0_s]
        else:
            iso_bar_0_s = np.arange(wsp_point[0].s+min_s,wsp_point[1].s+max_s,step_s).tolist()
            iso_bar_0_h = [WSP(P = wsp_point[1].P, s = i).h for i in iso_bar_0_s]
        plt.plot(iso_bar_0_s,iso_bar_0_h,color)

    #Диапазон h-s:
    def plot_hs_nozzle_t(x_lim, y_lim):
        plt.plot([point_0.s, point_1t.s], [point_0.h, point_1t.h], 'ro-')
        iso_bar(point_0, -0.02, 0.02, 0.001, 'c')
        iso_bar(point_1t, -0.02, 0.02, 0.001, 'y')
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    def callculate_optimum(d, p_0, T_0, n, G_0, H_0, rho, l_1, alpha_1, b_1, Delta, b_2, kappa_vs):
        u = M.pi*d*n
        point_0 = WSP(P = p_0, T = T_0)
        H_0s = H_0*(1-rho)
        H_0r = H_0*rho
        h_1t = point_0.h - H_0s
        point_1t = WSP(h = h_1t, s = point_0.s)
        c_1t = (2000*H_0s)**0.5
        M_1t = c_1t/point_1t.w
        mu_1 = 0.982 - 0.005*(b_1/l_1)
        F_1 = G_0*point_1t.v/mu_1/c_1t
        el_1 = F_1/M.pi/d/M.sin(M.radians(alpha_1))
        e_opt=5*el_1**0.5
        if e_opt > 0.85:
            e_opt = 0.85
        l_1 = el_1/e_opt
        fi_1 = 0.98 - 0.008*(b_1/l_1)
        c_1 = c_1t*fi_1
        alpha_1 = M.degrees(M.asin(mu_1/fi_1*M.sin(M.radians(alpha_1))))
        w_1 = (c_1**2+u**2-2*c_1*u*M.cos(M.radians(alpha_1)))**0.5
        betta_1 = M.degrees(M.atan(M.sin(M.radians(alpha_1))/(M.cos(M.radians(alpha_1))-u/c_1)))
        Delta_Hs = c_1t**2/2*(1-fi_1**2)
        h_1 = h_1t + Delta_Hs*1e-3
        point_1 = WSP(P = point_1t.P, h = h_1)
        h_2t = h_1 - H_0r
        point_2t = WSP(h = h_2t, s = point_1.s)
        w_2t = (2*H_0r*1e3+w_1**2)**0.5
        l_2 = l_1 + Delta
        mu_2 = 0.965-0.01*(b_2/l_2)
        M_2t = w_2t/point_2t.w
        F_2 = G_0*point_2t.v/mu_2/w_2t
        betta_2 = M.degrees(M.asin(F_2/(e_opt*M.pi*d*l_2)))
        point_1w = WSP(h = point_1.h+w_1**2/2*1e-3, s = point_1.s)
        psi = 0.96 - 0.014*(b_2/l_2)
        w_2 = psi*w_2t
        c_2 = (w_2**2+u**2-2*u*w_2*M.cos(M.radians(betta_2)))**0.5
        alpha_2 = M.degrees(M.atan(M.sin(M.radians(betta_2))/(M.cos(M.radians(betta_2))-u/w_2)))
        if alpha_2<0:
            alpha_2 = 180 + alpha_2
        Delta_Hr = w_2t**2/2*(1-psi**2)
        h_2 = h_2t+Delta_Hr*1e-3
        point_2 = WSP(P = point_2t.P, h = h_2)
        Delta_Hvs = c_2**2/2
        E_0 = H_0 - kappa_vs*Delta_Hvs
        etta_ol1 = (E_0*1e3 - Delta_Hs-Delta_Hr-(1-kappa_vs)*Delta_Hvs)/(E_0*1e3)
        etta_ol2 = (u*(c_1*M.cos(M.radians(alpha_1))+c_2*M.cos(M.radians(alpha_2))))/(E_0*1e3)
        return etta_ol2, alpha_2

    with st.container():
        left_column, right_column = st.columns([1,1.2])
        with left_column:
            H_0 = [i for i in list(range(90, 110, 1))]
            eta = []
            ucf = []
            graph1 = plt.figure()
            for i in H_0:
                ucf_1 = M.pi * d * n / (2000 * i) ** 0.5
                ucf.append(ucf_1)
                eta_ol, alpha = callculate_optimum(d, p_0, T_0, n, G_0, i, rho, l_1, alpha_1, b_1, delta, b_2, kappa_vs)
                eta.append(eta_ol)
            plt.grid(True)
            plt.xlabel("оптимальная скорость")
            plt.ylabel("КПД")
            plt.plot(ucf, eta)
            st.pyplot(graph1)

        with right_column:
            itog=pd.DataFrame({
            "H_0, кДж/кг": (list(np.arange(90,110,1))),
            "eta_oi": eta,
            "alpha": alpha,
            "ucf_1": ucf
            })
            itog

    H_0 = 90 #при этом значении кпд максимальный!
    u = M.pi*d*n
    point_0 = WSP(P = p_0, T = T_0)
    H_0s = H_0*(1-rho)
    H_0r = H_0*rho
    h_1t = point_0.h - H_0s
    point_1t = WSP(h = h_1t, s = point_0.s)
    c_1t = (2000*H_0s)**0.5
    M_1t = c_1t/point_1t.w
    mu_1 = 0.982 - 0.005*(b_1/l_1)
    F_1 = G_0*point_1t.v/mu_1/c_1t
    el_1 = F_1/M.pi/d/M.sin(M.radians(alpha_1))
    e_opt=6*el_1**0.5
    if e_opt > 0.85:
        e_opt = 0.85
    l_1 = el_1/e_opt

    print(f'l_1 = {l_1:.4f} м')
    if alpha_1 <= 10:
        NozzleBlade = 'C-90-09A'
        t1_ = 0.78
        b1_mod = 6.06
        f1_mod = 3.45
        W1_mod = 0.471
        alpha_inst1 = alpha_1-12.5*(t1_-0.75)+20.2
    elif  10 < alpha_1 <= 13:
        NozzleBlade = 'C-90-12A'
        t1_ = 0.78
        b1_mod = 5.25
        f1_mod = 4.09
        W1_mod = 0.575
        alpha_inst1 = alpha_1-10*(t1_-0.75)+21.2
    elif  13 < alpha_1 <= 16:
        NozzleBlade = 'C-90-15A'
        t1_ = 0.78
        b1_mod = 5.15
        f1_mod = 3.3
        W1_mod = 0.45
        alpha_inst1 = alpha_1-16*(t1_-0.75)+23.1
    else:
        NozzleBlade = 'C-90-18A'
        t1_ = 0.75
        b1_mod = 4.71
        f1_mod = 2.72
        W1_mod = 0.333
        alpha_inst1 = alpha_1-17.7*(t1_-0.75)+24.2
    print('Тип профиля:',  NozzleBlade)
    print(f'Оптимальный относительный шаг t1_ = {t1_}')
    z1 = (M.pi*d)/(b_1*t1_)
    z1 = int(z1)
    if z1 % 2 == 0:
        print(f'z1 = {z1}')
    else:
        z1 = z1+1
        print(f'z1 = {z1}')
    t1_ = (M.pi*d)/(b_1*z1)
    Ksi_1_ = (0.021042*b_1/l_1 + 0.023345)*100
    k_11 = 7.18977510*M_1t**5 - 26.94497258*M_1t**4 + 39.35681781*M_1t**3 - 26.09044664*M_1t**2 + 6.75424811*M_1t + 0.69896998
    k_12 = 0.00014166*90**2 - 0.03022881*90 + 2.61549380
    k_13 = 13.25474043*t1_**2 - 20.75439502*t1_ + 9.12762245
    Ksi_1 = Ksi_1_*k_11*k_12*k_13

    fi_1 = M.sqrt(1-Ksi_1/100)
    alpha_1 = 12
    c_1 = c_1t*fi_1
    alpha_1 = M.degrees(M.asin(mu_1/fi_1*M.sin(M.radians(alpha_1))))
    w_1 = (c_1**2+u**2-2*c_1*u*M.cos(M.radians(alpha_1)))**0.5
    c_1u = c_1*M.cos(M.radians(alpha_1))
    c_1a = c_1*M.sin(M.radians(alpha_1))
    w_1u = c_1u - u
    betta_1 = M.degrees(M.atan(M.sin(M.radians(alpha_1))/(M.cos(M.radians(alpha_1))-u/c_1)))
    Delta_Hs = c_1t**2/2*(1-fi_1**2)
    h_1 = h_1t + Delta_Hs*1e-3
    point_1 = WSP(P = point_1t.P, h = h_1)
    h_2t = h_1 - H_0r
    point_2t = WSP(h = h_2t, s = point_1.s)
    w_2t = (2*H_0r*1e3+w_1**2)**0.5
    l_2 = l_1 + delta
    mu_2 = 0.965-0.01*(b_2/l_2)
    M_2t = w_2t/point_2t.w
    F_2 = G_0*point_2t.v/mu_2/w_2t
    betta_2 = M.degrees(M.asin(F_2/(e_opt*M.pi*d*l_2)))
    point_1w = WSP(h = point_1.h+w_1**2/2*1e-3, s = point_1.s)


    if betta_2 <= 15:
        RotorBlade = 'P-23-14A'
        t2_ = 0.63
        b2_mod = 2.59
        f2_mod = 2.44
        W2_mod = 0.39
        beta_inst2 = betta_2-12.5*(t2_-0.75)+20.2
    elif  15 < betta_2 <= 19:
        RotorBlade = 'P-26-17A'
        t2_ = 0.65
        b2_mod = 2.57
        f2_mod = 2.07
        W2_mod = 0.225
        beta_inst2 = betta_2-19.3*(t2_-0.6)+60
    elif  19 < betta_2 <= 23:
        RotorBlade = 'P-30-21A'
        t2_ = 0.63
        b2_mod = 2.56
        f2_mod = 1.85
        W2_mod = 0.234
        beta_inst2 = betta_2-12.8*(t2_-0.65)+58
    elif 23 < betta_2 <= 27:
        RotorBlade = 'P-35-25A'
        t2_ = 0.6
        b2_mod = 2.54
        f2_mod = 1.62
        W2_mod = 0.168
        beta_inst2 = betta_2-16.6*(t2_-0.65)+54.3
    elif 27 < betta_2 <= 31:
        RotorBlade = 'P-46-29A'
        t2_ = 0.51
        b2_mod = 2.56
        f2_mod = 1.22
        W2_mod = 0.112
        beta_inst2 = betta_2-50.5*(t2_-0.6)+47.1
    else:
        RotorBlade = 'P-50-33A'
        t2_ = 0.49
        b2_mod = 2.56
        f2_mod = 1.02
        W2_mod = 0.079
        beta_inst2 = betta_2-20.8*(t2_-0.6)+43.7

    z2 = int((M.pi*d)/(b_2*t2_))
    t2_ = (M.pi*d)/(b_2*z2)
    Ksi_2_ = 4.364*b_2/l_2 + 4.22
    k_21 = -13.79438991*M_2t**4 + 36.69102267*M_2t**3 - 32.78234341*M_2t**2 + 10.61998662*M_2t + 0.28528786
    k_22 = 0.00331504*betta_1**2 - 0.21323910*betta_1 + 4.43127194
    k_23 = 60.72813684*t2_**2 - 76.38053189*t2_ + 24.97876023
    Ksi_2 = Ksi_2_*k_21*k_22*k_23
    psi = M.sqrt(1-Ksi_2/100)
    psi = 0.91
    w_2 = psi*w_2t
    c_2 = (w_2**2+u**2-2*u*w_2*M.cos(M.radians(betta_2)))**0.5
    alpha_2 = M.degrees(M.atan(M.sin(M.radians(betta_2))/(M.cos(M.radians(betta_2))-u/w_2)))
    Delta_Hr = w_2t**2/2*(1-psi**2)
    h_2 = h_2t+Delta_Hr*1e-3
    point_2 = WSP(P = point_2t.P, h = h_2)
    Delta_Hvs = c_2**2/2
    E_0 = H_0 - kappa_vs*Delta_Hvs
    point_3 = WSP(P = point_2t.P, h = (h_2+Delta_Hvs*1e-3))
    etta_ol1 = (E_0*1e3 - Delta_Hs-Delta_Hr-(1-kappa_vs)*Delta_Hvs)/(E_0*1e3)
    etta_ol2 = (u*(c_1*M.cos(M.radians(alpha_1))+c_2*M.cos(M.radians(alpha_2))))/(E_0*1e3)

    st.info("Приведенны графики для максимального КПД при H_0 = 90 кДж/кг")

    with st.container():
        left_column, right_column = st.columns([1,1])
        with left_column:
            graph2 = plt.figure()
            def plot_hs_stage_t(x_lim,y_lim):
                plot_hs_nozzle_t(x_lim,y_lim)
                plt.plot([point_0.s,point_1.s],[point_0.h,point_1.h],'bo-')
                plt.plot([point_1.s,point_2t.s],[point_1.h,point_2t.h], 'ro-')
                plt.plot([point_1.s,point_1.s],[point_1w.h, point_1.h],'ro-')
                plt.plot([point_1.s, point_2.s], [point_1.h, point_2.h], 'ro-')
                plt.plot([point_2.s, point_3.s], [point_2.h, point_3.h], 'ro-')
                iso_bar(point_2t,-0.02,0.02,0.001,'y')
                iso_bar(point_1w,-0.005,0.005,0.001,'c')
            plot_hs_stage_t([6.55,6.65],[3350,3480])
            plt.grid(True)
            plt.title("h - s диаграмма")
            plt.xlabel("s, кДж/(кг*К)")
            plt.ylabel("h, кДж/кг")
            st.pyplot(graph2)

            #Расчет скоростей:
            c_1u = c_1*M.cos(M.radians(alpha_1))
            c_1a = c_1*M.sin(M.radians(alpha_1))
            w_1u = c_1u - u
            w_2a = w_2*M.sin(M.radians(betta_2))
            w_2u = w_2*M.cos(M.radians(betta_2))
            c_2u=w_2u+u
            w_1_tr = [0, 0, -w_1u, -c_1a]
            c_1_tr = [0, 0, -c_1u, -c_1a]
            u_1_tr = [-w_1u, -c_1a, -u, 0]
        with right_column:
            graph3 = plt.figure()
            w_2_tr = [0, 0, w_2u, -w_2a]
            c_2_tr = [0, 0, c_2u, -w_2a]
            u_2_tr = [c_2u,-w_2a, -u, 0]
            ax = plt.axes()
            ax.arrow(*c_1_tr, head_width=5, length_includes_head = True,head_length=20, fc='r', ec='r')
            ax.arrow(*w_1_tr, head_width=5, length_includes_head = True,head_length=20, fc='b', ec='b')
            ax.arrow(*u_1_tr, head_width=5, length_includes_head = True,head_length=20, fc='g', ec='g')
            ax.arrow(*c_2_tr, head_width=5, length_includes_head = True,head_length=20, fc='r', ec='r')
            ax.arrow(*w_2_tr, head_width=5, length_includes_head = True,head_length=20, fc='b', ec='b')
            ax.arrow(*u_2_tr, head_width=5, length_includes_head = True,head_length=20, fc='g', ec='g')
            plt.text(-2*c_1u/3, -3*c_1a/4, '$c_1$', fontsize=20)
            plt.text(-2*w_1u/3, -3*c_1a/4, '$w_1$', fontsize=20)
            plt.text(2*c_2u/3, -3*w_2a/4, '$c_2$', fontsize=20)
            plt.text(2*w_2u/3, -3*w_2a/4, '$w_2$', fontsize=20)
            plt.grid(True)
            plt.title("Векторы скоростей:")
            st.pyplot(graph3)


    delta_a = 0.0025
    z_per_up = 2
    mu_a = 0.5
    mu_r = 0.75
    d_per = d + l_1
    delta_r = d_per*0.001
    delta_ekv = 1/M.sqrt(1/(mu_a*delta_a)**2+z_per_up/(mu_r*delta_r)**2)
    xi_u_b=M.pi*d_per*delta_ekv*etta_ol1/F_1*M.sqrt(rho+1.8*l_2/d)
    Delta_Hub = xi_u_b*E_0
    k_tr=0.0007
    Kappa_VS = 0
    u = M.pi*d*n
    c_f = M.sqrt(2000*H_0)
    ucf = u/c_f
    xi_tr = k_tr*d**2/F_1*ucf**3
    Delta_Htr = xi_tr*E_0
    k_v = 0.065
    m = 1
    xi_v = k_v/M.sin(M.radians(alpha_1))*(1-e_opt)/e_opt*ucf**3*m
    i_p = 4
    B_2 = b_2*M.sin(M.radians(beta_inst2))
    xi_segm = 0.25*B_2*l_2/F_1*ucf*etta_ol1*i_p
    xi_parc = xi_v+xi_segm
    Delta_H_parc = E_0*xi_parc

    H_i = E_0 - Delta_Hr*1e-3 - Delta_Hs*1e-3 - (1-Kappa_VS)*Delta_Hvs*1e-3 - Delta_Hub - Delta_Htr - Delta_H_parc
    eta_oi = H_i/E_0
    st.subheader("""Внутренний относительный КПД ступени  
            eta_oi  = %.3f """ % eta_oi)
    N_i = G_0*H_i
    st.subheader("""Внутреняя мощность ступени  
                N_i = = %.2f кВт""" % N_i)

if selected == "Задание 3":
    # Праметры точки торможения:
    p0 = 12.9
    t0 = 546
    T0 = t0 + 273.15
    point0 = WSP(P=p0, T=T0)
    h0 = point0.h
    p_tor = p0 - 0.05 * p0
    point_tor = WSP(P=p0, h=h0)
    h_tor = point_tor.h
    ppp = 2.48
    p_1t = ppp + 0.1 * ppp
    with st.container():
        st.subheader('Дано:')
        st.markdown('Диаметр регулирующей ступени: drc = 1.04 м')
        st.markdown("""Давление полного торможения перед нерегулируемой ступенью: P0 = """ + str(round(p_tor,3)) + """ МПа""")
        st.markdown("""Энтальпия полного торможения перед нерегулируемой ступенью: h0 = """ + str(round(h_tor,3)) + """ Дж/кг""")
        st.markdown("Частота вращения вала: n = 60 Гц")
        st.markdown("Расход пара в первую нерегулируемую ступень: G0 = 165.091 кг/c")
        st.markdown("""Давление пара за ЦВД: pz = """ + str(round(p_1t,3)) + """ МПа""")
        st.markdown("Количество ступеней ЦВД прототипа: Z = 8")
        st.subheader(' ')
        st.subheader('Корректировка данных: ')
        rho_s = st.slider('Степень реактивности первой нерегулируемой ступени в корне rho_s:', min_value=0.03, max_value=0.07, value=0.05, step=0.001)
        alfa = st.slider('Эффективный угол выхода потока из сопловой решетки: alfa', min_value=13, max_value=16, value=15, step=1)
        fi = st.slider('Коэффициент скорости сопловой решетки: fi', min_value=0.93, max_value=0.96, value=0.96, step=0.01)
        mu1 = st.slider('Коэффициент расхода сопловой решетки первой нерегулируемой ступени: mu1', min_value=0.95, max_value=0.97, value=0.97, step=0.01)
    drc = 1.04
    D1 = drc - 0.26
    h0 = h_tor
    p0 = p_tor
    pz = p_1t
    sat_steam = WSP(P=p0, h=h0)
    s_0 = sat_steam.s
    t_0 = sat_steam.T
    tetta = 20
    G0 = 165.091
    n = 60
    delta = 0.003
    etaoi = 0.88
    Z = 8
    error = 2
    i = 1

    error = 2
    i = 1
    while error > 0.5:
        rho = rho_s + 1.8 / (tetta + 1.8)
        X = (fi * M.cos(M.radians(alfa))) / (2 * M.sqrt(1 - rho))
        H01 = 12.3 * (D1 / X) ** 2 * (n / 50) ** 2
        h2t = h0 - H01
        steam2t = WSP(h=h2t, s=s_0)
        v2t = steam2t.v
        l11 = G0 * v2t * X / (M.pi ** 2 * D1 ** 2 * n * M.sqrt(1 - rho) * M.sin(M.radians(alfa)) * mu1)
        tetta_old = tetta
        tetta = D1 / l11
        error = abs(tetta - tetta_old) / tetta_old * 100
        i += 1

    l21 = l11 + delta
    d_s = D1 - l21
    steam_tz = WSP(P=pz, s=s_0)
    h_zt = steam_tz.h
    H0 = h0 - h_zt
    Hi = H0 * etaoi
    h_z = h0 - Hi
    steam_z = WSP(P=pz, h=h_z)
    v_2z = steam_z.v
    x = Symbol('x')
    с = solve(x ** 2 + x * d_s - (l21 * (d_s + l21) * v_2z / v2t))
    for j in с:
        if j > 0:
            l2z = j
    d2z = d_s + l2z
    tetta1 = (l21 + d_s) / l21
    tettaz = (l2z + d_s) / l2z
    rho1 = rho_s + 1.8 / (1.8 + tetta1)
    rhoz = rho_s + 1.8 / (1.8 + tettaz)
    X1 = (fi * M.cos(M.radians(alfa))) / (2 * M.sqrt(1 - rho1))
    Xz = (fi * M.cos(M.radians(alfa))) / (2 * M.sqrt(1 - rhoz))

    DeltaZ = 1
    ite = 0
    while DeltaZ > 0:
        matr = []
        Num = 0
        SumH = 0
        for _ in range(int(Z)):
            li = (l21 - l2z) / (1 - Z) * Num + l21
            di = (D1 - d2z) / (1 - Z) * Num + D1
            tetta_i = di / li
            rho_i = rho_s + 1.8 / (1.8 + tetta_i)
            X_i = (fi * M.cos(M.radians(alfa))) / (2 * M.sqrt(1 - rho_i))
            if Num < 1:
                H_i = 12.3 * (di / X_i) ** 2 * (n / 50) ** 2
            else:
                H_i = 12.3 * (di / X_i) ** 2 * (n / 50) ** 2 * 0.95
            Num = Num + 1
            H_d = 0
            SumH = SumH + H_i
            matr.append([Num, round(di, 3), round(li, 3), round(tetta_i, 2), round(rho_i, 3), round(X_i, 3), round(H_i, 2),round(H_d, 2)])
        H_m = SumH / Z
        q_t = 4.8 * 10 ** (-4) * (1 - etaoi) * H0 * (Z - 1) / Z
        Z_new = round(H0 * (1 + q_t) / H_m)
        DeltaZ = abs(Z - Z_new)
        Z = Z_new
        ite += 1
    DeltaH = (H0 * (1 + q_t) - SumH) / Z
    a = 0
    for elem in matr:
        matr[a][7] = round(elem[6]+DeltaH,2)
        a += 1

    N_=[]
    di_=[]
    li_=[]
    tettai_=[]
    rhoi_=[]
    Xi_=[]
    Hi_=[]
    Hdi_=[]
    a = 0
    for elem in matr:
        N_.append(matr[a][0])
        di_.append(matr[a][1])
        li_.append(matr[a][2])
        tettai_.append(matr[a][3])
        rhoi_.append(matr[a][4])
        Xi_.append(matr[a][5])
        Hi_.append(matr[a][6])
        Hdi_.append(matr[a][7])
        a += 1
    di_ = [float(x) for x in di_]
    li_ = [float(x) for x in li_]
    tettai_ = [float(x) for x in tettai_]
    rhoi_ = [float(x) for x in rhoi_]
    Xi_ = [float(x) for x in Xi_]
    Hi_ = [float(x) for x in Hi_]
    Hdi_ = [float(x) for x in Hdi_]

    st.info("Таблица 1. Распределение параметров по ступеням:")
    with st.container():
        itog=pd.DataFrame( {"№ ступени": (N_),
                               "di, м": (di_),
                               "li, м": (li_),
                               "θi ": (tettai_),
                               "ρi ": (rhoi_),
                               "Xi ": (Xi_),
                               "Hi, кДж/кг": (Hi_),
                               "Hi + Δ, кДж/кг": (Hdi_)
                               }
                           )
        st.dataframe(itog)
    z =[]
    for a in range(1, Z+1):
        z.append(a)

    st.write("#")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 15, 1))
    plt.grid(True)
    plt.plot(z, di_, '-ro')
    plt.title('Рисунок 1. Распределение средних диаметров по проточной части')
    st.pyplot(fig)

    st.write("#")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 15, 1))
    plt.grid(True)
    plt.plot(z, li_, '-ro')
    plt.title('Рисунок 2. Распределение высот лопаток по проточной части')
    st.pyplot(fig)

    st.write("#")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 15, 1))
    plt.grid(True)
    plt.plot(z, tettai_, '-ro')
    plt.title('Рисунок 3. Распределение обратной веерности по проточной части')
    st.pyplot(fig)

    st.write("#")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 15, 1))
    plt.grid(True)
    plt.plot(z, rhoi_, '-ro')
    plt.title('Рисунок 4. Распределение степени реактивности по проточной части')
    st.pyplot(fig)

    st.write("#")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 15, 1))
    plt.grid(True)
    plt.plot(z, Xi_, '-ro')
    plt.title('Рисунок 5. Распределение U/Cф по проточной части')
    st.pyplot(fig)

    st.write("#")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 15, 1))
    plt.grid(True)
    plt.plot(z, Hi_, '-ro')
    plt.title('Рисунок 6. Распределение теплоперепадов по проточной части')
    st.pyplot(fig)

    st.write("#")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 15, 1))
    plt.grid(True)
    plt.plot(z, Hdi_, '-ro')
    plt.title('Рисунок 7. Распределение теплоперепадов с учетом невязки по проточной части')
    st.pyplot(fig)
