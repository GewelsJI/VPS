- We greatly appreciate  [@Yuli Zhou](https://github.com/zhoustan) for the feedback and for fixing the sorting bug in our test and evaluation processes.

This bug fix has a very minor impact on performance. Below, we showcase the results before and after the fix:
|        Dataset         |      Method      |   Status       | Smeasure | meanEm | wFmeasure | maxDice |
|------------------------|------------------|----------------|----------|--------|-----------|---------|
| TestEasyDataset-Seen   | 2022-MIR-PNSPlus |   before-fix   |   0.917  |  0.924 |    0.848  |  0.888  |
| TestEasyDataset-Seen   | 2022-MIR-PNSPlus |   after-fix    |   0.917  |  0.924 |    0.848  |  0.888  |
| TestHardDataset-Seen   | 2022-MIR-PNSPlus |   before-fix   |   0.887  |  0.929 |    0.806  |  0.855  |
| TestHardDataset-Seen   | 2022-MIR-PNSPlus |   after-fix    |   0.887  |  0.902 |    0.806  |  0.855  |



|        Dataset         |      Method      |   Status       | Smeasure | meanEm | wFmeasure | maxDice | meanFm | meanSen |
|------------------------|------------------|----------------|----------|--------|-----------|---------|--------|---------|
| TestEasyDataset-Unseen | 2022-MIR-PNSPlus |   before-fix   |   0.806  |  0.798 |    0.676  |  0.756  |  0.730 |   0.630 |
| TestEasyDataset-Unseen | 2022-MIR-PNSPlus |   after-fix    |   0.806  |  0.798 |    0.676  |  0.756  |  0.730 |   0.630 |
| TestHardDataset-Unseen | 2022-MIR-PNSPlus |   before-fix   |   0.797  |  0.793 |    0.653  |  0.737  |  0.709 |   0.623 |
| TestHardDataset-Unseen | 2022-MIR-PNSPlus |   after-fix    |   0.798  |  0.793 |    0.654  |  0.737  |  0.709 |   0.624 |


Reminder: To ensure the correctness of results, we strongly recommend running the sorting strategy on a Linux system.