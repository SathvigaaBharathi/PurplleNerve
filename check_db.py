import asyncio
from app.db import AsyncSessionLocal
from sqlalchemy import text

async def check():
    async with AsyncSessionLocal() as db:
        r = await db.execute(text("SELECT COUNT(*) FROM events"))
        print('Total events:', r.scalar())
        
        r2 = await db.execute(text("""
            SELECT store_id, event_type, timestamp, visitor_id, zone_id, is_staff, confidence 
            FROM events 
            ORDER BY timestamp DESC 
            LIMIT 15
        """))
        rows = r2.fetchall()
        print('\nLatest 15 events:')
        for row in rows:
            print(f"  {row[0]} | {row[1]:20s} | {str(row[2])[:19]} | {row[3][:15]} | zone={row[4]} | staff={row[5]} | conf={row[6]:.2f}")
        
        r3 = await db.execute(text("""
            SELECT 
                COUNT(DISTINCT visitor_id) FILTER (WHERE is_staff = false AND event_type = 'ENTRY') as unique_visitors,
                DATE(timestamp) as event_date
            FROM events 
            WHERE store_id = 'STORE_BLR_002'
            GROUP BY DATE(timestamp)
            ORDER BY event_date DESC
            LIMIT 5
        """))
        print('\nVisitor counts by date:')
        for row in r3.fetchall():
            print(f"  date={row[1]} unique_visitors={row[0]}")

asyncio.run(check())
