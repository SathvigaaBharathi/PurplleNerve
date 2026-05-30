import csv
import os
import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, update, text, insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import DBPosTransaction, DBEvent, DBSessionConversion

logger = logging.getLogger(__name__)

def parse_iso_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp format, replacing 'Z' with UTC timezone info."""
    ts_str = ts_str.strip()
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    return datetime.fromisoformat(ts_str)

async def load_pos_transactions_from_csv(csv_path: str, db: AsyncSession):
    """Loads transactions from a CSV file and inserts them into the DB."""
    if not os.path.exists(csv_path):
        logger.warning(f"POS CSV file not found at: {csv_path}")
        return 0
        
    try:
        inserted_count = 0
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Normalize column names in case they have spaces or spelling variations
            fieldnames = [fn.strip() for fn in reader.fieldnames] if reader.fieldnames else []
            
            transactions = []
            for row in reader:
                # Clean keys
                row_cleaned = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                
                txn_id = row_cleaned.get("transaction_id") or row_cleaned.get("transaction")
                store_id = row_cleaned.get("store_id") or row_cleaned.get("store")
                ts_str = row_cleaned.get("timestamp") or row_cleaned.get("time")
                basket_val = row_cleaned.get("basket_value_inr") or row_cleaned.get("basket_value") or row_cleaned.get("value")
                
                if not (txn_id and store_id and ts_str and basket_val):
                    continue
                    
                transactions.append({
                    "transaction_id": txn_id,
                    "store_id": store_id,
                    "timestamp": parse_iso_timestamp(ts_str),
                    "basket_value": float(basket_val)
                })

            if transactions:
                # Bulk insert transactions
                stmt = pg_insert(DBPosTransaction).values(transactions)
                stmt = stmt.on_conflict_do_nothing(index_elements=["transaction_id"])
                res = await db.execute(stmt)
                await db.commit()
                inserted_count = len(transactions)
                logger.info(f"Successfully loaded {inserted_count} POS transactions from CSV.")
                
        return inserted_count
    except Exception as e:
        logger.error(f"Error loading POS transactions CSV: {e}")
        await db.rollback()
        return 0

async def correlate_transactions(db: AsyncSession):
    """
    Correlates POS transactions to visitor sessions.
    Runs every 60 seconds in the background.
    """
    try:
        # 1. Fetch all transactions that haven't been matched yet (or all to re-evaluate)
        stmt_txn = select(DBPosTransaction) # we can query all for reliability, or just matched_visitor == None
        res_txn = await db.execute(stmt_txn)
        transactions = res_txn.scalars().all()
        
        correlated_count = 0
        
        for txn in transactions:
            # Look back 5 minutes from transaction timestamp
            window_start = txn.timestamp - timedelta(minutes=5)
            window_end = txn.timestamp
            
            # Query non-staff visitors in BILLING / BILLING_COUNTER / CASHIER zones during window
            stmt_visitor = select(DBEvent.visitor_id).where(
                DBEvent.store_id == txn.store_id,
                DBEvent.zone_id.in_(["BILLING", "BILLING_COUNTER", "CASHIER"]),
                DBEvent.is_staff == False,
                DBEvent.timestamp >= window_start,
                DBEvent.timestamp <= window_end
            ).distinct()
            
            res_visitor = await db.execute(stmt_visitor)
            visitor_ids = res_visitor.scalars().all()
            
            if visitor_ids:
                # Mark as converted in conversions table (insert all matched visitors)
                conv_inserts = []
                for vid in visitor_ids:
                    conv_inserts.append({
                        "store_id": txn.store_id,
                        "visitor_id": vid,
                        "converted_at": datetime.now(timezone.utc)
                    })
                
                # Bulk insert conversions (ON CONFLICT DO NOTHING)
                stmt_conv = pg_insert(DBSessionConversion).values(conv_inserts)
                stmt_conv = stmt_conv.on_conflict_do_nothing(index_elements=["store_id", "visitor_id"])
                await db.execute(stmt_conv)
                
                # Update matched_visitor in pos_transactions table with first match
                stmt_update = update(DBPosTransaction).where(
                    DBPosTransaction.transaction_id == txn.transaction_id
                ).values(matched_visitor=visitor_ids[0])
                await db.execute(stmt_update)
                
                correlated_count += 1
                
        if correlated_count > 0:
            await db.commit()
            logger.info(f"Correlated {correlated_count} POS transactions to visitor sessions.")
    except Exception as e:
        logger.error(f"Error during POS correlation task: {e}")
        await db.rollback()
