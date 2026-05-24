from prometheus_client import Counter, Histogram

upload_counter = Counter("receipt_upload_total", "Total receipt uploads processed")
pipeline_latency_seconds = Histogram("receipt_pipeline_latency_seconds", "Latency for full Day-1 sync pipeline")
