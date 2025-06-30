import json
import signal
import time
from typing import List, Dict, Tuple

from redis import Redis, exceptions
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 导入文本分割器

from app.configs.config import settings
from app.rag.rag_service import get_embeddings
from app.utils.singleton import chroma_collection, redis_pool, logger

SHUTDOWN_REQUESTED = False

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "，", "、", " "],
    length_function=len
)


def handle_shutdown(signum, frame):
    """停机信号处理器"""
    global SHUTDOWN_REQUESTED
    if not SHUTDOWN_REQUESTED:
        logger.warning("Shutdown signal received. Finishing current batch before exiting...")
        SHUTDOWN_REQUESTED = True


def sanitize_metadata_value(value):
    """【工具函数】如果值是列表或字典，将其JSON序列化为字符串。"""
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        except TypeError:
            return str(value)
    return value


def process_messages_batch(messages: List[Tuple[str, Dict[str, str]]]):
    """
    一次性处理一批消息。
    对每条消息中的大文档进行切分，然后将所有切分出的文本块统一进行向量化和存储。
    """
    first_msg_id = messages[0][0]
    last_msg_id = messages[-1][0]
    logger.debug(f"Parsing batch of {len(messages)} messages from {first_msg_id} to {last_msg_id}.")

    # 这些列表现在将包含切分后的所有文本块（chunks）的数据
    batch_ids, batch_embeddings, batch_metadatas, batch_documents = [], [], [], []

    for msg_id, msg_data in messages:
        try:
            data = json.loads(msg_data['data'])
            # 这里我们仍然使用 source_id 作为文档的唯一标识符
            source_id = "dynamic::" + data.get('source_id')
            content = data.get('content')

            if not all([source_id, content]):
                raise ValueError("Message is missing 'source_id' or 'content'")

            # 1. 对文档内容进行切分
            chunks = text_splitter.split_text(content)
            logger.debug(f"Document {source_id} was split into {len(chunks)} chunks.")

            # 2. 为每个切分出的文本块准备数据
            sanitized_metadata = {key: sanitize_metadata_value(value) for key, value in
                                  data.get('metadata', {}).items()}

            for i, chunk_content in enumerate(chunks):
                # 为每个 chunk 创建一个唯一的 ID，格式为 "原始ID::chunk::序号"
                # 这使得每个 chunk 都有唯一ID，同时保持了与源文档的关联性。
                chunk_id = f"{source_id}::chunk::{i}"

                batch_ids.append(chunk_id)
                batch_documents.append(chunk_content)
                # 所有来自同一源文档的 chunk 共享相同的元数据
                batch_metadatas.append(sanitized_metadata)

        except Exception as e:
            logger.error(f"Failed to parse or chunk message. Error: {e}", extra={"msg_id": msg_id})

    # 如果所有消息都处理失败（例如都是空内容），则直接返回
    if not batch_documents:
        # ACK这批消息以防止“毒丸消息”阻塞队列
        all_ids_to_ack = [msg_id for msg_id, _ in messages]
        return all_ids_to_ack

    try:
        # 批量向量化所有切分出的文本块
        batch_embeddings = get_embeddings(batch_documents)

        # 批量写入向量数据库
        chroma_collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )
        logger.info(f"Successfully processed and upserted {len(batch_ids)} chunks from {len(messages)} messages.")

        # 返回所有被成功处理的原始消息ID，以便ACK
        return [msg_id for msg_id, _ in messages]

    except Exception as e:
        logger.error(f"Failed to process batch embeddings/upsert. Error: {e}", extra={"msg_id": "batch_operation"})
        return []  # 批量处理失败，这批消息都算失败，不进行ACK


def run_sync_worker():
    """主运行循环"""
    r = Redis(connection_pool=redis_pool)
    try:
        r.xgroup_create(
            name=settings.REDIS_STREAM_NAME,
            groupname=settings.REDIS_CONSUMER_GROUP_NAME,
            id='0-0',
            mkstream=True
        )
    except exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    while not SHUTDOWN_REQUESTED:
        try:
            messages = r.xreadgroup(
                groupname=settings.REDIS_CONSUMER_GROUP_NAME,
                consumername=settings.REDIS_CONSUMER_NAME,
                streams={settings.REDIS_STREAM_NAME: '>'},
                count=settings.REDIS_MESSAGES_PER_PULL,
                block=settings.REDIS_BLOCK_TIMEOUT_MS
            )
            if not messages:
                continue

            message_list = messages[0][1]
            # 批量处理消息
            successfully_processed_original_msg_ids = process_messages_batch(message_list)

            # 只ACK成功处理的原始消息ID
            if successfully_processed_original_msg_ids:
                r.xack(settings.REDIS_STREAM_NAME, settings.REDIS_CONSUMER_GROUP_NAME,
                       *successfully_processed_original_msg_ids)
        except exceptions.ConnectionError as e:
            logger.error(f"Redis connection error: {e}. Retrying in 5 seconds...", extra={'msg_id': 'N/A'})
            time.sleep(5)
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}", extra={'msg_id': 'N/A'})
            time.sleep(5)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    run_sync_worker()
