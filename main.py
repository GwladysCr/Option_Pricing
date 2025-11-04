#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:04:20 2025

@author: gwladyscrenn
"""
from European_options import OptionPricingTool

tool = OptionPricingTool(ticker='AAPL', option_type='call')
results = tool.run_analysis()
tool.plot_results()
