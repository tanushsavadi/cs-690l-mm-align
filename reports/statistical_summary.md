# Statistical Evidence Summary

This file is generated from cached artifacts only. No model was rerun.

- Baseline run: `2026-04-08-standard_dpo-pilot-7`
- Candidate run: `2026-04-08-image_aware_dpo-pilot-7`

## Bootstrap Metric Deltas

| source | benchmark | metric | n | baseline_value | candidate_value | delta | ci_low | ci_high | prob_candidate_higher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark | chartqa | relaxed_accuracy | 1920 | 0.3797 | 0.4062 | 0.0266 | 0.0177 | 0.0359 | 1.0000 |
| benchmark | hallusionbench | accuracy | 1129 | 0.5722 | 0.5740 | 0.0018 | -0.0080 | 0.0124 | 0.6230 |
| benchmark | pope | accuracy | 9000 | 0.8733 | 0.8718 | -0.0016 | -0.0032 | 0.0003 | 0.0410 |
| benchmark | pope | f1 | 9000 | 0.8830 | 0.8814 | -0.0016 | -0.0033 | 0.0000 | 0.0260 |

The delta column is `image_aware_dpo - standard_dpo`. A confidence interval crossing zero means the result should be described as a small or uncertain effect, not a clean win.

## Dependence Deltas

| source | benchmark | metric | n | baseline_value | candidate_value | delta | ci_low | ci_high | prob_candidate_higher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dependence | all | blank_changed_rate | 12049 | 0.9667 | 0.9705 | 0.0037 | 0.0017 | 0.0060 | 1.0000 |
| dependence | all | mismatch_changed_rate | 12049 | 0.9285 | 0.9361 | 0.0076 | 0.0041 | 0.0110 | 1.0000 |
| dependence | all | blank_score_drop_mean | 12049 | 0.4272 | 0.4200 | -0.0071 | -0.0110 | -0.0030 | 0.0010 |
| dependence | all | mismatch_score_drop_mean | 12049 | 0.3508 | 0.3508 | 0.0000 | -0.0030 | 0.0028 | 0.4960 |

These rows measure how much answers change when images are blanked or mismatched. Higher changed rates suggest more visual sensitivity, but score-drop metrics still need to be interpreted carefully.

## Paired Case Counts

| benchmark | case_type | count | rate | total |
| --- | --- | --- | --- | --- |
| chartqa | image_aware_only_correct | 64 | 0.0333 | 1920 |
| chartqa | standard_only_correct | 13 | 0.0068 | 1920 |
| chartqa | both_correct | 716 | 0.3729 | 1920 |
| chartqa | both_wrong | 1127 | 0.5870 | 1920 |
| hallusionbench | image_aware_only_correct | 16 | 0.0142 | 1129 |
| hallusionbench | standard_only_correct | 14 | 0.0124 | 1129 |
| hallusionbench | both_correct | 632 | 0.5598 | 1129 |
| hallusionbench | both_wrong | 467 | 0.4136 | 1129 |
| pope | image_aware_only_correct | 27 | 0.0030 | 9000 |
| pope | standard_only_correct | 41 | 0.0046 | 9000 |
| pope | both_correct | 7819 | 0.8688 | 9000 |
| pope | both_wrong | 1113 | 0.1237 | 9000 |

## Representative Case Types

| case_type | benchmark | sample_id | prompt | ground_truth | prediction_baseline | prediction_candidate | is_correct_baseline | is_correct_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_aware_win | chartqa | chartqa-val-augmented-102 | How many dollars were the gate receipts of the Chicago Bulls in the 2019/20 season? | 65 | The gate receipts for the Chicago Bulls in the 2019/20 season were $65 million. | 65 | False | True |
| image_aware_win | chartqa | chartqa-val-augmented-105 | What percentage of Bed Bath & Beyond's global sales did home furnishings account for in fiscal year 2020? | 65.3 | In fiscal year 2020, home furnishings accounted for 65 percent of Bed Bath & Beyond's global sales. | 65 | False | True |
| image_aware_win | chartqa | chartqa-val-augmented-142 | What percentage of students were offered a full-time job after completing an unpaid internship in the Government? | 31 | The bar chart shows that 45% of students who completed an unpaid internship in the government were offered a full-time job after their internship. | 31 | False | True |
| image_aware_win | chartqa | chartqa-val-augmented-146 | What was the prevalence of asthma in Spain in 2015? | 1.58 | The prevalence of asthma in Spain in 2015 was 1.58 million individuals. | 1.58 | False | True |
| image_aware_win | chartqa | chartqa-val-augmented-148 | How many stores did Nordstrom operate as of January 30, 2021? | 369 | As of January 30, 2021, Nordstrom operated 369 stores. | 369 | False | True |
| image_aware_win | chartqa | chartqa-val-augmented-182 | How many international trips were made on land in 2019? | 5.25 | According to the chart, in 2019 there were approximately 5.25 million international trips made on land. | 5.25 | False | True |
| image_aware_win | chartqa | chartqa-val-augmented-183 | What was Nexon's revenue in 2020? | 293 | The revenue of Nexon in 2020 was 293 billion yen. | 293 | False | True |
| image_aware_win | chartqa | chartqa-val-augmented-187 | What was the number of active subscribers for ALBtelecom in 2015? | 648079 | The number of active subscribers for ALBtelecom in 2015 was 648,079. | 648079 | False | True |
| standard_win | chartqa | chartqa-val-augmented-125 | How many women were in Greenland in 2021? | 13394 | According to the bar chart, there were 13,394 women living in Greenland in 2021. | According to the bar chart, in 2021 there were 13,394 women living in Greenland. | True | False |
| standard_win | chartqa | chartqa-val-augmented-129 | What percentage of 18 to 29 year olds said they would cancel Netflix? | 49 | According to the chart, 49% of 18-29 year olds said they would cancel Netflix if it were free. | According to the chart, 40% of 18-29 year olds said they would cancel Netflix if it were free. | True | False |
| standard_win | chartqa | chartqa-val-augmented-140 | How much money are consumers expected to spend over the course of the Black Friday weekend in 2020? | 7.5 | The bar chart shows that consumers are expected to spend approximately 7.5 billion GBP on Black Friday in 2020, with this figure being the total spending across both online and offline channels. | In 2020, consumers were expected to spend around 5.76 billion GBP on online shopping during the Black Friday weekend. | True | False |
| standard_win | chartqa | chartqa-val-augmented-200 | What was Conair's market share for beard and mustache trimmers in 2008? | 30 | According to the chart, Conair's market share for beard and mustache trimmers was 30% in 2008. | The market share for Conair's beard and mustache trimmers in 2008 was 30%. | True | False |
| standard_win | chartqa | chartqa-val-augmented-45 | What percentage of potash reserves did Canada have in 2019? | 27.8 | According to the pie chart, Canada had approximately 27.8% of the global potash reserves in 2019. | In 2019, Canada had approximately 27.8 percent of the world's potash reserves. | True | False |
| standard_win | chartqa | chartqa-val-augmented-556 | How much did Katrina's insured losses amount to? | 82.39 | The insured losses from Hurricane Katrina amounted to 82.39 billion U.S. dollars. This figure represents the total financial impact of the hurricane on the insurance industry, including direct damage to property and infrastructure, as well as indirect losses such as business interruption and loss of revenue. | The insured losses from Hurricane Katrina, which occurred in 2005 and caused significant damage to oil rigs in the Gulf of Mexico, amounted to $82.39 billion. This makes it the most expensive natural disaster event globally based on insured losses according to the provided data. | True | False |
| standard_win | chartqa | chartqa-val-augmented-582 | How many visitors did Yosemite National Park see in 2020? | 2.27 | According to the provided data, Yosemite National Park saw approximately 2.27 million visitors in 2020. | In 2020, Yosemite National Park saw approximately 2.27 million visitors. | True | False |
| standard_win | chartqa | chartqa-val-augmented-751 | What percentage of online sales is expected to increase to by 2018? | 0.7 | The expected increase in the share of online sales is 0.7 percent, as shown in the bar graph for the year 2018. | The chart shows that the share of online sales is expected to increase from 0.4% in 2013 to 0.7% in 2018. Therefore, the percentage of online sales is expected to increase to **0.7%** by 2018. | True | False |
| both_fail | chartqa | chartqa-val-augmented-0 | Who portrayed Jon Snow? | Kit Harington | Based on the information provided in the chart, Kit Harington portrayed Jon Snow. The chart shows his net worth as $12 million and his earnings per episode as $0.5 million. | Based on the information provided in the bar graph, Kit Harington portrayed Jon Snow. The graph shows Kit Harington's net worth as $12 million and his earnings per episode as $0.5 million. | False | False |
| both_fail | chartqa | chartqa-val-augmented-1 | Who played Mother of Dragons Daenerys Targaryen? | Emilia Clarke | According to the chart, Emilia Clarke played the character Daenerys Targaryen in the TV series "Game of Thrones." | According to the chart, Emilia Clarke played the role of Daenerys Targaryen in "Game of Thrones." | False | False |
| both_fail | chartqa | chartqa-val-augmented-10 | What was the average ticket in general merchandisers and grocers during the week of February 25 to March 2, 2020? | 53 | The average ticket in general merchandisers during the week of February 25 to March 2, 2020, was $46, while the average ticket in grocers was $53. | The average ticket for general merchandisers during the week of February 25 to March 2, 2020, was $46, while the average ticket for grocers was $53. | False | False |
| both_fail | chartqa | chartqa-val-augmented-101 | How many people applied for universal credit in the two weeks leading up to January 9, 2020? | 94015 | According to the graph provided by Statista, the number of new claimants applying for Universal Credit in the two weeks leading up to January 9, 2020 was 94,015. | The number of new claimants applying for Universal Credit in the two weeks leading up to January 9, 2020, was 94,015. | False | False |

## Short Interpretation

The strongest result is still ChartQA, where image-aware DPO is higher. HallusionBench is basically a tie with a tiny image-aware edge. POPE slightly favors standard DPO. The honest conclusion is that image-aware DPO gives modest grounding-related gains, not a universal improvement.
