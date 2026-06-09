



def main() -> None:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing UN Comtrade / UNSD HS mappings")
    print(f"Raw directory:        {RAW_DIR}")
    print(f"Normalized directory: {NORMALIZED_DIR}")
    print()

    for spec in MAPPING_SPECS:
        raw_path = RAW_DIR / spec.raw_filename

        print("=" * 72)
        print(f"Input:  {raw_path.name}")
        print(
            "Output direction: "
            f"{spec.normalized_source_system} -> {spec.normalized_target_system}"
        )
        print(f"Reverse direction: {spec.reverse_direction}")

        raw_df = read_relationship_sheet(raw_path, spec)
        normalized = normalize_mapping(raw_df, spec)

        out_path = NORMALIZED_DIR / output_filename(spec)
        normalized.to_csv(out_path, index=False)

        relationship_counts = (
            normalized["Relationship"]
            .value_counts()
            .rename_axis("Relationship")
            .reset_index(name="n_rows")
            .sort_values("Relationship")
        )

        print(f"Rows written: {len(normalized):,}")
        print(f"Output: {out_path}")
        print("Relationship counts:")
        print(relationship_counts.to_string(index=False))
        print()

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()