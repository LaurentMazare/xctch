import argparse
from executorch.devtools import Inspector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--etdump", type=str)
    parser.add_argument("--etrecord", type=str)
    args = parser.parse_args()
    etrecord_path = args.etrecord
    etdump_path = args.etdump
    inspector = Inspector(etdump_path=etdump_path, etrecord=etrecord_path)
    
    for event_block in inspector.event_blocks:
        orig_df = event_block.to_dataframe()
        print(orig_df.iloc[:2].T)
    print(orig_df.shape)
    df = orig_df[orig_df["event_name"] != "OPERATOR_CALL"]
    print(df["event_name"].value_counts())
    
    print(df["avg"].sum())
    print(">>>>")
    _df = df[df["event_name"] == "Method::execute"]
    print(_df.T)
    
    _df = df[["event_name", "avg"]]
    _df = (
        _df.groupby('event_name')
        .agg(
            avg=('avg', 'sum'),
            rows=('avg', 'count')
        )
        .sort_values(by='avg', ascending=False)
        .reset_index()
    )
    
    print(_df)
    
    _df = df.sort_values(by='avg', ascending=False).reset_index()
    print(_df.iloc[:50][["event_name", "avg", "module_hierarchy", "stack_traces"]])
    print(_df.iloc[50:100][["event_name", "avg", "module_hierarchy", "stack_traces"]])
    
    # _df = df[df["event_name"] == "native_call_linear.out"]
    # print(_df.iloc[:1].T)
    
    _df = orig_df[orig_df["event_name"] == "OPERATOR_CALL"]
    print("OPERATOR CALL", _df.shape)
    _df = _df.sort_values(by='avg', ascending=False)
    _df = _df.drop(columns=["raw"])
    print(_df.iloc[:5].T)
    
    _df = _df[_df["module_hierarchy"].apply(bool)]
    print("WITH HIERARCHY", _df.shape, _df["avg"].sum())
    print(_df[["module_hierarchy", "avg"]])
    print(_df.iloc[0]["stack_traces"])

if __name__ == "__main__":
    main()
