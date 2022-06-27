#%%
from pickle import FALSE
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np
import streamlit as stream
import scipy.stats as st
import math

stream.set_page_config(
     page_title="Case S&OP",
     page_icon="🎲",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )


#%%

def apply_debito(df, qtd_):
    debt_ = qtd_
    k = 0
    while(debt_>0 and k<len(df)):

        if(df.iloc[k,1]>=debt_):
            df.iloc[k,1] = df.iloc[k,1] - debt_
            debt_ = 0
        elif(df.iloc[k,1]<debt_):
            df.iloc[k,1] = 0
            debt_ = debt_ - df.iloc[k,1]

        k = k+1
    return debt_, df

#%%

def generate_forecast_(MEAN_DEMAND = 100, STD_DEMAND = 50, SIZE_FORECAST = 1000, MAPE_OFFSET = 0.3, MAPE = 1):

    #CREATE FORECAST
    MEAN_DEMAND = 100
    STD_DEMAND = 50
    SIZE_FORECAST = 1000

    day_vec = []
    for i in range(0,SIZE_FORECAST):
        day_ = date.today() + timedelta(days=i)
        day_vec.append(day_)

    data = np.random.normal(MEAN_DEMAND,STD_DEMAND,SIZE_FORECAST)

    dataframe_forecast = pd.DataFrame(columns=['data','forecast'])
    dataframe_forecast['data'] = day_vec
    dataframe_forecast['forecast'] = data

    dataframe_forecast['forecast'] = [0 if x<=0 else int(x) for x in dataframe_forecast['forecast']]

    real_sales = dataframe_forecast.copy()
    real_sales['sales'] = real_sales['forecast']
    real_sales = real_sales.drop('forecast', axis=1)

    real_sales['sales'] = [int(x*(1+np.random.normal(MAPE_OFFSET, MAPE, 1))) for x in real_sales['sales']]
    real_sales['sales'] = [0 if x<=0 else int(x) for x in real_sales['sales']]

    return dataframe_forecast, real_sales

#%%

def get_cashflow_and_stock_sim(stock_map_dataframe, pay_dataframe, TIME_RANGE = 240, flag_backtest=True):

    CASH_FLOW_VEC = []
    STOCK_VEC = []

    CASH_BANK = 0
    COUNT_FIRST_SALE = 0
    for i in range(0, TIME_RANGE):

        day_sim = date.today()+timedelta(days=i)

        if(flag_backtest):
            sale_  = dataframe_forecast[dataframe_forecast['data']==day_sim].forecast.sum()
        else:
            sale_  = real_sales[real_sales['data']==day_sim].sales.sum()

        qtd_estoque = stock_map_dataframe[stock_map_dataframe['data']<=day_sim].lote.sum()

        if(COUNT_FIRST_SALE>=0 and COUNT_FIRST_SALE<LTREC_CLIENTE and qtd_estoque>0):
            COUNT_FIRST_SALE = COUNT_FIRST_SALE + 1
            #stock_map_dataframe = apply_debito(stock_map_dataframe, AVG_SALE_MULT)[1]
            stock_map_dataframe = apply_debito(stock_map_dataframe, sale_)[1]


        if(COUNT_FIRST_SALE>=LTREC_CLIENTE and qtd_estoque>0):
            #CASH_BANK = CASH_BANK + AVG_SALE_MULT*PRICE
            #stock_map_dataframe = apply_debito(stock_map_dataframe, AVG_SALE_MULT)[1]
            stock_map_dataframe = apply_debito(stock_map_dataframe, sale_)[1]
            CASH_BANK = CASH_BANK + sale_*PRICE


        stock_map_dataframe = stock_map_dataframe[stock_map_dataframe['lote']>0]

        DEBT_ = -1*pay_dataframe[pay_dataframe['data']==day_sim].valor_pay.sum()
        CASH_BANK = CASH_BANK + DEBT_

        qtd_estoque = stock_map_dataframe[stock_map_dataframe['data']<=day_sim].lote.sum()

        STOCK_VEC.append([day_sim, qtd_estoque])
        CASH_FLOW_VEC.append([day_sim, CASH_BANK])

        if(len(stock_map_dataframe)==0 and flag_backtest):
            break

    dataframe_stock_simulation = pd.DataFrame(data = STOCK_VEC, columns = ['data_sim','pos_estoque'])
    dataframe_cashflow_simulation = pd.DataFrame(data = CASH_FLOW_VEC, columns = ['data_sim','cash_flow'])

    return dataframe_stock_simulation, dataframe_cashflow_simulation

#%%

def position_order(pay_dataframe, stock_map_dataframe, order_to_be, order_date_, lerror1, lerror2):

    PAYMENT_DATE_VEC_FUT = [[order_date_+timedelta(days=LTPAG2+(LT1+lerror1)+(LT2+lerror2)), order_to_be*COST_OF_LABOR],
                [order_date_+timedelta(days=LTPAG1+LT1), order_to_be*COST_OF_GOODS]
                ]

    pay_dataframe = pay_dataframe.append(pd.DataFrame(data = PAYMENT_DATE_VEC_FUT, columns = ['data','valor_pay']))

    STOCK_MAP_FUT = [[order_date_+timedelta(days=(LT1+lerror1)+(LT2+lerror2)), order_to_be]]
    stock_map_dataframe = stock_map_dataframe.append(pd.DataFrame(data = STOCK_MAP_FUT, columns = ['data','lote']))

    return stock_map_dataframe, pay_dataframe

#%%
#PARAMETERS FROM SUPPLIERS

stream.markdown("### 🎲 Case S&OP - Thiago Villani | Supply Chain Analysis | Insider")

with stream.expander("Parâmetros de Supply Chain"):
    stream.write("""
        Variáveis associadas aos prazos da cadeia
    """)

    LT1 = stream.slider('LT1', min_value=0, max_value=100, value=30, step = 10, help='LeadTime de Entrega da matéria prima')
    LT2 = stream.slider('LT2', min_value=0, max_value=100, value=30, step = 10, help='LeadTime de Entrega da Confeccção')
    LTREC_CLIENTE = stream.slider('LTREC CLIENTE', min_value=0, max_value=30, value=3, step = 1, help='LeadTime de Recebimento do Pagamento do Cliente')
    LTPAG2 = stream.slider('LTPAG2', min_value=0, max_value=120, value=30, step = 10, help='Prazo de Pagamento da Confecção')
    LTPAG1 = stream.slider('LTPAG1', min_value=0, max_value=120, value=90, step = 10, help='Prazo de Pagamento da Matéria Prima')

#PARAMETERS OF SUPPLY

SIZE_FORECAST = 1000
LEADTIME_ERROR = 0

with stream.expander("Parâmetros de Demanda"):
    stream.write("""
        Variáveis associadas a demanda estimada e o modelo de previsão
    """)

    STOCK_MAX_ = stream.slider('STOCK_MAX_', min_value=0, max_value=120, value=60, step = 10, help='Cobertura máxima de estoque')
    MEAN_DEMAND = stream.slider('MEAN_DEMAND', min_value=0, max_value=200, value=100, step = 10, help='Giro médio em unidades de produtos')
    STD_DEMAND = stream.slider('STD_DEMAND', min_value=0, max_value=100, value=50, step = 10, help='Desvio padrão da saída média')

    MAPE_OFFSET = stream.slider('MAPE_OFFSET', min_value=-1.0, max_value=1.0, value=0.0, step = 0.1, help='Viés de previsão de demanda')
    MAPE = stream.slider('MAPE', min_value=-1.0, max_value=1.0, value=0.0, step = 0.1, help='Erro de previsão de demanda')

    SERVICE_LEVEL = stream.slider('SERVICE LEVEL', min_value=0, max_value=99, value=50, step = 1, help='Nível de Serviço desejado')

#PARAMETERS OF FINANCE
with stream.expander("Parâmetros de Financeiros"):
    stream.write("""
        Variáveis associadas aos parâmetros financeiros da operação
    """)

    PRICE = stream.slider('PRICE', min_value=5, max_value=50, value=10, step = 5, help='Preço médio das unidades')
    MARKUP = stream.slider('MARKUP', min_value=0.0, max_value=5.0, value=3.5, step = 0.5, help='Markup dos produtos')
    COST_OF_GOODS_PERC = stream.slider('% COST RM', min_value=0.0, max_value=1.0, value=0.7, step = 0.1, help='% do custo de matéria prima no custo total')

TOTAL_COST = PRICE/MARKUP
COST_OF_GOODS = COST_OF_GOODS_PERC*TOTAL_COST
COST_OF_LABOR = (1-COST_OF_GOODS_PERC)*TOTAL_COST

#PARAMETERS OF SIMULATION
N_MAX_ORDERS = 10
TIME_RANGE = 1000

dataframe_forecast, real_sales = generate_forecast_(MEAN_DEMAND = MEAN_DEMAND,
                                                     STD_DEMAND = STD_DEMAND,
                                                      SIZE_FORECAST = SIZE_FORECAST,
                                                         MAPE_OFFSET = MAPE_OFFSET,
                                                             MAPE = MAPE)

ORDER = dataframe_forecast[(dataframe_forecast['data']>=pd.to_datetime(date.today()+timedelta(days=LT2+LT1))) &
 (dataframe_forecast['data']<=pd.to_datetime(date.today()+timedelta(days=LT2+LT1+STOCK_MAX_)))].forecast.sum()

PAYMENT_DATE_VEC = [[date.today()+timedelta(days=LTPAG2+LT1+LT2), ORDER*COST_OF_LABOR],
                    [date.today()+timedelta(days=LTPAG1+LT1), ORDER*COST_OF_GOODS]
]

pay_dataframe = pd.DataFrame(data = PAYMENT_DATE_VEC, columns = ['data','valor_pay'])

STOCK_MAP = [[date.today()+timedelta(days=LT2+LT1), ORDER]]
stock_map_dataframe = pd.DataFrame(data = STOCK_MAP, columns = ['data','lote'])

ORDER_BACKLOG_VEC = []

ZETA = st.norm.ppf(SERVICE_LEVEL/100)
SAFETY_STOCK = ZETA*math.sqrt(LT1+LT2)*STD_DEMAND# + (LT1+LT2)*MEAN_DEMAND

n_orders_ = 0
while(n_orders_<=N_MAX_ORDERS):
    print("At order n_ : {}".format(n_orders_))
    dataframe_stock_simulation, dataframe_cashflow_simulation  = get_cashflow_and_stock_sim(stock_map_dataframe, pay_dataframe, TIME_RANGE = TIME_RANGE, flag_backtest=True)

    first_rupture_dataframe = dataframe_stock_simulation[(dataframe_stock_simulation['data_sim']>=date.today()+timedelta(days=LT1+LT2)) &
    (dataframe_stock_simulation['pos_estoque']<=SAFETY_STOCK)]

    #stream.write(first_rupture_dataframe.head(40))

    if(len(first_rupture_dataframe)>0):
        data_ruptura = first_rupture_dataframe.head(1).data_sim.tolist()[0]
        order_date_ = data_ruptura - timedelta(days=LT1+LT2)

        new_order_ = dataframe_forecast[(dataframe_forecast['data']>=pd.to_datetime(order_date_+timedelta(days=LT2+LT1))) &
        (dataframe_forecast['data']<=pd.to_datetime(order_date_+timedelta(days=LT2+LT1+STOCK_MAX_)))].forecast.sum()

        ld_order_error1 = int(np.random.normal(0, LEADTIME_ERROR, 1))
        ld_order_error2 = int(np.random.normal(0, LEADTIME_ERROR, 1))

        stock_map_dataframe, pay_dataframe =  position_order(pay_dataframe, stock_map_dataframe, new_order_, order_date_, ld_order_error1, ld_order_error2)

        real_sales_backlog = real_sales[(real_sales['data']>=pd.to_datetime(order_date_+timedelta(days=LT2+LT1))) &
        (real_sales['data']<=pd.to_datetime(order_date_+timedelta(days=LT2+LT1+STOCK_MAX_)))].sales.sum()

        demand_mape_error = abs(new_order_ - real_sales_backlog)/real_sales_backlog

        ORDER_BACKLOG_VEC.append([order_date_, new_order_, ld_order_error1, ld_order_error2, demand_mape_error])

        n_orders_ = n_orders_ + 1
    else:
        break

DF_ORDER = pd.DataFrame(data = ORDER_BACKLOG_VEC, columns = ['order_date_', 'new_order_', 'ld_order_error1', 'ld_order_error2', 'demand_mape_error'])

stream.write("""
        Sugestões de Pedidos
    """)
stream.write(DF_ORDER)

#%%

dataframe_stock_simulation, dataframe_cashflow_simulation  = get_cashflow_and_stock_sim(stock_map_dataframe, pay_dataframe, TIME_RANGE = TIME_RANGE, flag_backtest=False)

dataframe_stock_simulation = dataframe_stock_simulation[dataframe_stock_simulation['data_sim']<=pd.to_datetime(datetime(2023,12,31))]
dataframe_cashflow_simulation = dataframe_cashflow_simulation[dataframe_cashflow_simulation['data_sim']<=pd.to_datetime(datetime(2023,12,31))]

#%%


valid_range = dataframe_stock_simulation[(dataframe_stock_simulation['data_sim']>=dataframe_stock_simulation[dataframe_stock_simulation['pos_estoque']>0].head(1).data_sim.tolist()[0]) &
(dataframe_stock_simulation['data_sim']<=dataframe_stock_simulation[dataframe_stock_simulation['pos_estoque']>0].tail(1).data_sim.tolist()[0])]

#%%

indicador_disponibilidade = 1 - len(valid_range[valid_range['pos_estoque']==0])/len(valid_range)
cashflow_final = dataframe_cashflow_simulation.tail(1).cash_flow.tolist()[0]
cashflow_max = dataframe_cashflow_simulation.cash_flow.max()
cashflow_min = dataframe_cashflow_simulation.cash_flow.min()

dataframe_cashflow_simulation_cp = dataframe_cashflow_simulation.reset_index()
cond = dataframe_cashflow_simulation_cp['cash_flow']<0
dataframe_cashflow_simulation_cp.date=pd.to_datetime(dataframe_cashflow_simulation_cp.data_sim)

max_time_negative_cashflow = dataframe_cashflow_simulation_cp[~cond].groupby(cond.cumsum())['data_sim'].agg(np.ptp).reset_index().sort_values(by='data_sim', ascending=False).cash_flow.tolist()[0]

#%%


col1, col2, col3 = stream.columns(3)
col1.metric("Disponibilidade", "{0:.1f}%".format(indicador_disponibilidade*100))
col2.metric("Cashflow Final", "R$ {0:.1f}k".format(cashflow_final/1000))
col3.metric("Duração Cashflow Negativo", "{} Dias".format(max_time_negative_cashflow))

dataframe_stock_simulation = dataframe_stock_simulation[dataframe_stock_simulation['data_sim']<=dataframe_stock_simulation[dataframe_stock_simulation['pos_estoque']>0].tail(1).data_sim.tolist()[0]]
dataframe_cashflow_simulation = dataframe_cashflow_simulation[dataframe_cashflow_simulation['data_sim']<=dataframe_stock_simulation[dataframe_stock_simulation['pos_estoque']>0].tail(1).data_sim.tolist()[0]]

# Create traces
fig = go.Figure()

fig.add_trace(go.Scatter(x=dataframe_stock_simulation['data_sim'],
                         y=dataframe_stock_simulation['pos_estoque'],
                         name='STOCK',
                    mode='lines+markers',
                    yaxis='y2'))



fig.add_trace(go.Scatter(x=dataframe_cashflow_simulation['data_sim'],
                         y=dataframe_cashflow_simulation['cash_flow'],
                         name='CASH FLOW',
                    mode='lines+markers'))


fig.add_trace(go.Scatter(x=dataframe_stock_simulation['data_sim'],
                         y=[SAFETY_STOCK]*len(dataframe_cashflow_simulation['data_sim']),
                         name='SAFETY STOCK',
                         line_dash='dashdot',
                    mode='lines+markers',
                    yaxis='y2'))


fig.update_layout(
    height=500,
    width = 1200,
    yaxis=dict(
        title="CASHFLOW",
        titlefont=dict(
            color="#1f77b4"
        ),
        tickfont=dict(
            color="#1f77b4"
        )
    ),
    yaxis2=dict(
        title="STOCK",
        titlefont=dict(
            color="#ff7f0e"
        ),
        tickfont=dict(
            color="#ff7f0e"
        ),
        overlaying="y",
        side="right"
    ))

stream.plotly_chart(fig, use_container_width=True)


# %%
