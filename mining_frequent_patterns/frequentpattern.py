import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


class FrequentPattern:
    def apriori_algorithm(self, dataset, min_support, confidence):
        min_support /= len(dataset)
        confidence /= 100

        te = TransactionEncoder()
        one_hot_boolean_arr = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(one_hot_boolean_arr, columns=te.columns_)
        frequent_dataset = apriori(df, min_support=min_support, use_colnames=True)

        rules = association_rules(frequent_dataset, metric='confidence', min_threshold=confidence)

        return rules[['antecedents', 'consequents']]


    def fp_growth_algorithm(self, dataset, min_support, confidence):
        min_support /= len(dataset)
        confidence /= 100
        te = TransactionEncoder()
        one_hot_boolean_arr = te.fit_transform(dataset)

        df = pd.DataFrame(one_hot_boolean_arr, columns=te.columns_)

        frequent_dataset = fpgrowth(df, min_support=min_support, use_colnames=True)

        rules = association_rules(frequent_dataset, metric='confidence', min_threshold=confidence)

        return rules[['antecedents', 'consequents']]


pattern = FrequentPattern()

# start
# for Apriori
dataset = [
    ['I1', 'I2', 'I5'],
    ['I2', 'I4'],
    ['I2', 'I3'],
    ['I1', 'I2', 'I4'],
    ['I1', 'I3'],
    ['I2', 'I3'],
    ['I1', 'I3'],
    ['I1', 'I2', 'I3', 'I5'],
    ['I1', 'I2', 'I3']
]



print("Apriori Algorithm: ")
print(pattern.apriori_algorithm(dataset, 2, 75))


# for FP Growth
dataset = [
    list('MONKEY'),
    list('DONKEY'),
    list('MAKE'),
    list('MUCKY'),
    list('COOKIE')
]

print("\n\nFP Growth Algorithm: ")
print(pattern.fp_growth_algorithm(dataset, 3, 80))
