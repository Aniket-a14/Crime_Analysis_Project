// Static imports for 100% reliability on Vercel
import data05 from '../../public/outputs/data_05.json';
import data06 from '../../public/outputs/data_06.json';
import data01 from '../../public/outputs/data_01.json';
import ledger from '../../public/outputs/ledger.json';
import csvRisk from '../../public/outputs/csv_risk.json';
import csvTrends from '../../public/outputs/csv_trends.json';

export async function getPythonOutputs() {
  try {
    // These are bundled at build time
    return {
      global: data01,
      ledger: ledger,
      trends: data02,
      severity: data03,
      hotspots: data04,
      forecasts: data05,
      riskMatrix: data06,
      csvRisk,
      csvTrends
    };
  } catch (error) {
    console.error('Error bundling python outputs:', error);
    return { error: 'Failed to build analytics context.' };
  }
}
