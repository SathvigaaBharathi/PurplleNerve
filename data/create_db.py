import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        conn = await asyncpg.connect('postgresql://postgres:postgres@localhost:5432/postgres')
        try:
            await conn.execute('CREATE DATABASE retail_intelligence')
            logger.info("Successfully created database 'retail_intelligence'")
        except asyncpg.DuplicateDatabaseError:
            logger.info("Database 'retail_intelligence' already exists")
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Failed to connect or create database: {e}")

if __name__ == "__main__":
    asyncio.run(main())
