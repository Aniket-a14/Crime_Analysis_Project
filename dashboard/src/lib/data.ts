import fs from 'fs/promises';
import path from 'path';

export async function getPythonOutputs() {
  try {
    const outputsDir = path.join(process.cwd(), 'public', 'outputs');
    
    const readJson = async (filename: string) => {
      try {
        const fileData = await fs.readFile(path.join(outputsDir, filename), 'utf-8');
        return JSON.parse(fileData);
      } catch (e) {
        console.error(`Failed to read ${filename}`, e);
        return null;
      }
    };

    const data02 = await readJson('data_02.json');
    const data03 = await readJson('data_03.json');
    const data04 = await readJson('data_04.json');
    const data05 = await readJson('data_05.json');
    const data06 = await readJson('data_06.json');

    return {
      trends: data02,
      severity: data03,
      hotspots: data04,
      forecasts: data05,
      riskMatrix: data06
    };
  } catch (error) {
    console.error('Error reading python outputs:', error);
    return { error: 'Failed to build analytics context.' };
  }
}
