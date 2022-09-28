from prepare import text_to_int, clean
from machine_learn import clf
import pandas as pd

test = pd.read_csv('/Users/nikitaushanov/Downloads/credit_test.csv')
test_ids = test["Loan ID"]

test = test.drop(["Loan ID", "Customer ID"], axis=1)
text_to_int(test, type_df='test')
clean(test)

submission_preds = clf.predict(test)

itog = pd.DataFrame({"Loan ID": test_ids.values,
                   "Loan Status": submission_preds,
                  })

itog.loc[(itog["Loan Status"] == 1), "Loan Status"] = "Fully Paid"
itog.loc[(itog["Loan Status"] == 0), "Loan Status"] = "Charged Off"

itog.to_csv("/Users/nikitaushanov/Downloads/itog.csv", index=False)