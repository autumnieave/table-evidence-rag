-- 为业务查询创建只读账号。执行前请替换 CHANGE_ME_STRONG_PASSWORD。
-- 注意：'%' 允许来自 Docker 网桥等远端地址；生产环境应改为明确的网段/主机。

CREATE USER IF NOT EXISTS 'rag_reader'@'localhost' IDENTIFIED BY 'CHANGE_ME_STRONG_PASSWORD';
CREATE USER IF NOT EXISTS 'rag_reader'@'%' IDENTIFIED BY 'CHANGE_ME_STRONG_PASSWORD';

GRANT SELECT ON financial_database.core_performance_indicators_sheet TO 'rag_reader'@'localhost';
GRANT SELECT ON financial_database.balance_sheet TO 'rag_reader'@'localhost';
GRANT SELECT ON financial_database.cash_flow_sheet TO 'rag_reader'@'localhost';
GRANT SELECT ON financial_database.income_sheet TO 'rag_reader'@'localhost';

GRANT SELECT ON financial_database.core_performance_indicators_sheet TO 'rag_reader'@'%';
GRANT SELECT ON financial_database.balance_sheet TO 'rag_reader'@'%';
GRANT SELECT ON financial_database.cash_flow_sheet TO 'rag_reader'@'%';
GRANT SELECT ON financial_database.income_sheet TO 'rag_reader'@'%';

FLUSH PRIVILEGES;
