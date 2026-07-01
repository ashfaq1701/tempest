#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

std::vector<std::tuple<int, int, int64_t>>
load_edges_from_parquet(const char* path);

#ifdef TRW_HAS_PARQUET

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#define TRW_ARROW_OK(expr) do {                                         \
    const auto _st = (expr);                                            \
    if (!_st.ok()) {                                                    \
        throw std::runtime_error(                                       \
            std::string("arrow error (" #expr "): ") + _st.ToString()); \
    }                                                                   \
} while (0)

std::vector<std::tuple<int, int, int64_t>>
load_edges_from_parquet(const char* path) {
    const std::string p(path);
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    auto infile_result = arrow::io::ReadableFile::Open(p, pool);
    if (!infile_result.ok()) {
        throw std::runtime_error("failed to open " + p + ": "
            + infile_result.status().ToString());
    }
    std::shared_ptr<arrow::io::ReadableFile> infile = *infile_result;

    auto reader_result = parquet::arrow::OpenFile(infile, pool);
    if (!reader_result.ok()) {
        throw std::runtime_error("parquet open(" + p + "): "
            + reader_result.status().ToString());
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_result);

    std::shared_ptr<arrow::Schema> schema;
    TRW_ARROW_OK(reader->GetSchema(&schema));

    std::vector<int> col_indices;
    for (const char* name : {"u", "i", "ts"}) {
        const int idx = schema->GetFieldIndex(name);
        if (idx < 0) {
            throw std::runtime_error(
                std::string("column '") + name + "' missing in " + p);
        }
        col_indices.push_back(idx);
    }

    std::shared_ptr<arrow::Table> table;
    TRW_ARROW_OK(reader->ReadTable(col_indices, &table));

    std::vector<std::tuple<int, int, int64_t>> out;
    out.reserve(static_cast<size_t>(table->num_rows()));

    const auto u_col  = table->column(0);
    const auto i_col  = table->column(1);
    const auto ts_col = table->column(2);

    // alibaba dumps u/i as int64; narrowing to int32
    for (int c = 0; c < u_col->num_chunks(); ++c) {
        const auto u_arr  = std::static_pointer_cast<arrow::Int64Array>(u_col->chunk(c));
        const auto i_arr  = std::static_pointer_cast<arrow::Int64Array>(i_col->chunk(c));
        const auto ts_arr = std::static_pointer_cast<arrow::Int64Array>(ts_col->chunk(c));
        const int64_t* u_raw  = u_arr->raw_values();
        const int64_t* i_raw  = i_arr->raw_values();
        const int64_t* ts_raw = ts_arr->raw_values();
        const int64_t len = u_arr->length();
        for (int64_t k = 0; k < len; ++k) {
            out.emplace_back(static_cast<int>(u_raw[k]),
                             static_cast<int>(i_raw[k]),
                             ts_raw[k]);
        }
    }
    return out;
}

#else  // !TRW_HAS_PARQUET

std::vector<std::tuple<int, int, int64_t>>
load_edges_from_parquet(const char* path) {
    (void)path;
    throw std::runtime_error(
        "parquet support not compiled in (pyarrow not found at configure time)");
}

#endif
