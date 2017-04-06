from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4)

suits = ['H', 'S', 'C', 'D']
card_val = (range(1, 11) + [10] * 3) * 4
base_names = ['A'] + range(2, 11) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)

deck = Series(card_val, index=cards)

def draw(deck, n=5):
    return deck.take(np.random.permutation(len(deck))[:n])
draw(deck)

def draw2(deck, n=5):
    return deck.take(np.random.permutation(10))
    
get_suit = lambda card: card[-1] # last letter is suit
deck.groupby(get_suit).apply(draw, n=2)

# alternatively
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)

x = abs(np.random.randn(8))
y = abs(np.random.randn(8))

df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'data': x,
                'weights': y})
                
grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
G = grouped.apply(get_wavg)

