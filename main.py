from matplotlib import markers
import numpy as np
from datetime import datetime
import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
from secrets import API_KEY_ID, SECRET_KEY
from pandas.core.frame import DataFrame
import mplfinance as mpf
import csv
import os

class Bot:
    api = alpaca.REST(API_KEY_ID, SECRET_KEY)
    sym = ''
    slow_period = 40
    fast_period = 15
    stock_data = pd.DataFrame()
    transactions = pd.DataFrame({
        "price": [],
        "buy": [],
        "balance": [],
        "volume": []
    }, index=[])
    state = 0
    balance_history = pd.DataFrame()
    init_stock_closing = -1

    def __init__(self, sym, initial_balance):
        self.transactions.index.name = 'timestamp'
        self.sym = sym
        self.initial_balance = initial_balance
        self.funds = initial_balance
        self.min_balance = initial_balance
        self.min_funds = initial_balance
        self.holding = {
            'close': 0,
            'volume': 0,
            'timestamp': '',
        }
        now = datetime.now()
        self.init_datetime = now
        self.datadir =  "data/{0}_{1}-{2}-{3}".format(self.sym, now.year, now.month, now.day)


    def get_balance(self):
        return self.funds + (self.holding['close'] * self.holding['volume'])

    def add_to_stockdata(self, data):
        if len(self.stock_data.index) > self.slow_period*4:
            self.stock_data = self.stock_data.tail(self.slow_period*4).append(data)
        else:
            self.stock_data = self.stock_data.append(data)

    def add_to_transactions(self, order):
        if len(self.transactions.index) > self.slow_period*4:
            self.transactions = self.transactions.tail(self.slow_period*4).append(order)
        else:
            self.transactions = self.transactions.append(order)
        
    def is_holding(self):
        if self.holding['volume'] > 0:
            return True
        return False

    def print_order(self, order):
        order_type = '[BUY]' if order['buy'].values == 1 else '[SELL]'
        datetimeind = pd.DatetimeIndex(order.index)[0]
        order['datetime_pretty'] = '{0}/{1}/{2} {3}:{4}'.format(
            datetimeind.month,
            datetimeind.day,
            datetimeind.year,
            datetimeind.hour,
            datetimeind.minute
        )
        print(order_type + ' {0} - {1}x {2} at price ${3}'.format(order['datetime_pretty'].values[0], order['volume'].values[0], self.sym, order['price'].values[0] // 0.01 / 100))

    def get_transactions_filename(self):
        return '{4}/{0}_transactions_{1}-{2}-{3}.csv'.format(self.sym, self.init_datetime.year, self.init_datetime.month, self.init_datetime.day, self.datadir)

    def save_transaction(self, order):
        file = open(self.get_transactions_filename(), 'a')
        writer = csv.writer(file)
        writer.writerow([
            order.index.values[0], 
            order['buy'].values[0], 
            order['balance'].values[0], 
            order['price'].values[0], 
            order['volume'].values[0], 
        ])
        file.close()
        return
    
    def get_all_transactions(self):
        filename = self.get_transactions_filename()
        df = pd.DataFrame()
        try:
            f = open(filename, "r")
        except:
            # no transactions were made
            return pd.DataFrame({
                'buy': [], 
                'balance': [], 
                'price': [], 
                'volume': [], 
            }, index=[])
        csvFile = csv.reader(f)
        for line in csvFile:
            row = pd.DataFrame({
                'buy': int(line[1]), 
                'balance': float(line[2]), 
                'price': float(line[3]), 
                'volume': float(line[4]), 
            }, index=[line[0]])
            df = df.append(row)
        f.close()
        return df

    def buy(self, shares, data):
        if self.is_holding():
            print('[ERROR] Cannot buy. Already holding {0}'.format(self.sym))
            return
        
        self.funds = self.funds - (shares * data['close'])
        self.holding['close'] = data['close']
        self.holding['volume'] = shares
        self.holding['timestamp'] = data.name
        buy_order = DataFrame([[1, self.get_balance(), data['close'], shares]], columns=['buy', 'balance', 'price', 'volume'], index=[data.name])
        self.add_to_transactions(buy_order)
        self.save_transaction(buy_order)
        self.print_order(buy_order)

    def sell(self, shares, data):
        if not self.is_holding():
            print("[ERROR] Cannot buy. Not holding any {0}".format(self.sym))

        self.funds = self.funds + (shares * data['close'])
        self.holding['close'] = 0
        self.holding['volume'] = 0
        self.holding['timestamp'] = ''

        sell_order = DataFrame([[0, self.get_balance(), data['close'], shares]], columns=['buy', 'balance', 'price', 'volume'], index=[data.name])
        self.add_to_transactions(sell_order)
        self.save_transaction(sell_order)
        self.print_order(sell_order)
    
    def tema_strategy(self):
        last_two = self.stock_data.tail(2)
        current = last_two.tail(1)
        previous = last_two.head(1)
        current_price = current['close'].values[0]

        # derivatives = pd.DataFrame()
        # derivatives['dTEMA'] = (self.stock_data['TEMA'].shift(periods=-1) - self.stock_data['TEMA'].shift(periods=1)) / (self.stock_data['TEMA'].shift(periods=-1) + self.stock_data['TEMA'].shift(periods=1))

        # if abs(derivatives.loc[previous.index]['dTEMA'].values[0] <= 10**(-7)) and not self.is_holding():
        #     return 1
        # elif abs(derivatives.loc[previous.index]['dTEMA'].values[0] <= 10**(-7)) and self.is_holding():
        #     return 0
        # return -1

        if current['TEMA'].values > current_price and not self.is_holding():
            return 1
        elif current['TEMA'].values < current_price and self.is_holding():
            return 0
        return -1

    def ewa_strategy(self):
        last_two = self.stock_data.tail(2)
        current = last_two.tail(1)
        previous = last_two.head(1)
        if previous['fast_EWA'].values < previous['slow_EWA'].values and current['fast_EWA'].values > current['slow_EWA'].values:
            return 0
        elif previous['fast_EWA'].values > previous['slow_EWA'].values and current['fast_EWA'].values < current['slow_EWA'].values:
            return 1
        return -1
    
    def make_decision(self, data):
        if self.init_stock_closing == -1:
            self.init_stock_closing = data['close']
        # Update holding
        if self.is_holding():
            self.holding['close'] = data['close']

        # Append data and calulate EWA
        self.add_to_stockdata(data)
        self.stock_data['slow_EWA'] = self.stock_data['close'].rolling(window=self.slow_period).mean()
        self.stock_data.loc[:self.slow_period, 'slow_EWA'] = np.nan
        self.stock_data['fast_EWA'] = self.stock_data['close'].ewm(span=self.fast_period, adjust=True).mean()
        self.stock_data.loc[:self.fast_period, 'fast_EWA'] = np.nan

        emas = pd.DataFrame()
        emas['EMA'] = self.stock_data['close'].rolling(window=10).mean()
        emas['EMA_1'] = emas['EMA'].rolling(window=10).mean()
        emas['EMA_2'] = emas['EMA_1'].rolling(window=10).mean()
        emas['EMA_3'] = emas['EMA_2'].rolling(window=10).mean()
        self.stock_data['TEMA'] = (3*emas["EMA_1"]) - (3*emas['EMA_2']) + emas['EMA_3']

        # Check it was appended
        current_price = self.stock_data.loc[data.name]['close']
        if not current_price == data['close']:
            print('Could not add data to list')
            return

        tema_res = self.tema_strategy()
        ewa_res = self.ewa_strategy()
        should_buy = -1
        if tema_res == ewa_res:
            should_buy = tema_res

        self.save_stock_record(self.stock_data.tail(1))
        # BUY
        if should_buy == 1 and not self.is_holding():
            if data['close'] >= self.funds:
                return
            shares = self.funds // data['close']
            self.buy(shares, data)
        # SELL
        elif should_buy == 0 and self.is_holding():
            bought_price = self.transactions.tail(1)['price'].values[0]
            shares = self.transactions.tail(1)['volume'].values[0]
            if data['close'] > bought_price*(1 + 0.02) or data['close'] <= bought_price*0.98:
                self.sell(shares, data)

        self.balance_history = self.balance_history.append(pd.DataFrame({
            "balance": self.get_balance(),
        }, index=[data.name]))

        if self.get_balance() < self.min_balance:
            self.min_balance = self.get_balance()
        if self.funds < self.min_funds:
            self.min_funds = self.funds

    def print_account(self):
        print("----- Account Information -----")
        print('Initial balance: ${0}'.format(self.initial_balance))
        print('Current funds: ${0}'.format(round(self.funds, 2)))
        print('Current balance: ${0}'.format(round(self.get_balance(), 2)))
        if self.is_holding():
            print('Total assets: {0}x {1} at ${2} (${3})'.format(self.holding['volume'], self.sym, self.holding['close'], self.holding['volume']*self.holding['close'] // 0.01 / 100))
        diff = (self.get_balance() - self.initial_balance)
        if  diff < 0:
            print('Account is down ${0} (%{1})'.format(diff // 0.01 / 100, (diff / self.initial_balance) // 0.01 / 100))
        elif diff > 0:
            print('Account is up ${0} (%{1})'.format(diff // 0.01 / 100, (diff / self.initial_balance) // 0.01 / 100 ))
        print("Minimum balance (%{0}): ${1}".format(round(self.min_balance / self.initial_balance * 100, 2), round(self.min_balance, 2)))    
        print("Minimum funds (%{0}): ${1}".format(round(self.min_funds / self.initial_balance * 100, 2), round(self.min_funds, 2)))   
    
    def run_test(self, start_date, end_date, timeframe):
        print("Getting data from alpaca...")
        data = self.api.get_bars(self.sym, timeframe, start_date, end_date).df
        data = data.filter(['close', 'open', 'volume', 'high', 'low'])
        print("Data received.")

        if not os.path.isdir(self.datadir):
            os.mkdir(self.datadir)
        if os.path.exists(self.get_stock_filename()):
            os.remove(self.get_stock_filename())
        if os.path.exists(self.get_transactions_filename()):
            os.remove(self.get_transactions_filename())
            
        start = datetime.now()
        for t, row in data.iterrows():
            self.make_decision(row)
        end = datetime.now()
        transactions = self.get_all_transactions()

        print('Elapsed time: {0}s'.format((end-start).total_seconds()))
        print('{0} trades were made'.format(len(transactions.index)))
        print()
        self.print_account()
       
        try:
            diff = bot.stock_data.tail(1)['close'][0] - bot.init_stock_closing
            diff = diff // 0.01 / 100
            init_volume = bot.transactions.head(1)['volume'][0]
            print("Net gain {0}: ${1} (${2})".format(self.sym, diff, diff*init_volume // 0.01 / 100))
        except:
            print("No transactions were made")

        print("Total decision duration: {0}".format(self.total_decision_time))
        return
    
    def get_all_stockdata(self):
        filename = self.get_stock_filename()
        df = pd.DataFrame()
        f = open(filename, "r")
        csvFile = csv.reader(f)
        for line in csvFile:
            row = pd.DataFrame({
                'close': float(line[1]), 
                'open': float(line[2]), 
                'high': float(line[3]), 
                'low': float(line[4]), 
                'volume': float(line[5]), 
                'slow_EWA': float(line[6]), 
                'fast_EWA': float(line[7]),
                'TEMA': float(line[8])
            }, index=[line[0]])
            df = df.append(row)
        f.close()
        return df

    def plot(self, candle = False):
        stock_data = self.get_all_stockdata()
        transactions = self.get_all_transactions()
        s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size': 6})
        balance = mpf.make_addplot(self.balance_history,ylabel='Balance', width=2, panel=0)
        slow = mpf.make_addplot(stock_data['slow_EWA'], panel=1, width=0.5, color='purple')
        fast = mpf.make_addplot(stock_data['fast_EWA'], panel=1, width=0.5, linestyle='--', color='purple')
        close = mpf.make_addplot(stock_data['close'], panel=1, width=1, color='blue')

        tema = mpf.make_addplot(stock_data['TEMA'], panel=1, width=1, color='pink')
        
        temp = stock_data.filter(['close'])
        temp.index = pd.to_datetime(temp.index).tz_localize('Etc/UCT')
        buy_orders = transactions[transactions['buy'] == 1]
        buy_orders.index = pd.to_datetime(buy_orders.index)

        buys = temp.merge(buy_orders, how='outer', left_index=True, right_index=True)
        bought = mpf.make_addplot(buys['price'], panel=1, color='green', width=5, scatter=True, marker='^')
        
        sell_orders = transactions[transactions['buy'] == 0]
        sells = temp.merge(sell_orders, how='outer', left_index=True, right_index=True)
        sold = mpf.make_addplot(sells['price'], panel=1, color='red', width=5, scatter=True, marker='v')
        stock_data.index = pd.to_datetime(stock_data.index)

        type = None
        if candle:
            type = 'candle'
        else:
            type = 'line'
        figure, axes = mpf.plot(stock_data, type=type, panel_ratios=(0.3, 1, 0.3),
            volume=True,
            title=self.sym,
            style=s,
            figsize=(10,5),
            returnfig=True,
            addplot=[balance,slow,fast, bought, sold, close, tema],
            main_panel=1,
            volume_panel=2,
            num_panels=3
        )

        for t, b in buys.iterrows():
            axes[1].text(b.name, b.price, "${0}".format(b.price), color='black', zorder=9)

        axes[0].yaxis.set_label_position("left")
        axes[0].yaxis.tick_left()
        axes[4].yaxis.set_label_position("left")
        axes[4].yaxis.tick_left()
        mpf.show()

    def get_stock_filename(self):
        return '{4}/{0}_stockdata_{1}-{2}-{3}.csv'.format(self.sym, self.init_datetime.year, self.init_datetime.month, self.init_datetime.day, self.datadir)

    def save_stock_record(self, data):
        filename = self.get_stock_filename()
        file = open(filename, 'a')
        writer = csv.writer(file)
        writer.writerow([
            data.index.values[0], 
            data['close'].values[0], 
            data['open'].values[0], 
            data['high'].values[0], 
            data['low'].values[0], 
            data['volume'].values[0], 
            data['slow_EWA'].values[0], 
            data['fast_EWA'].values[0],
            data['TEMA'].values[0]
        ])
        file.close()
        return

start_date = "2020-01-01"
end_date = "2020-01-30"
timeframe = TimeFrame.Minute

bot = Bot("F", 1000)
bot.run_test(start_date, end_date, timeframe)
stock_data = bot.get_all_stockdata()
#bot.plot()
