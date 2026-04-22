import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import Papa from 'papaparse';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const parentDir = path.join(process.cwd(), '..');
    const riskScoresPath = path.join(parentDir, 'public_safety_risk_scores.csv');
    const trendsPath = path.join(parentDir, 'rising_crime_trends.csv');

    const riskScoresCsv = await fs.readFile(riskScoresPath, 'utf-8');
    const trendsCsv = await fs.readFile(trendsPath, 'utf-8');

    const parsedRiskScores = Papa.parse(riskScoresCsv, { header: true, skipEmptyLines: true }).data;
    const parsedTrends = Papa.parse(trendsCsv, { header: true, skipEmptyLines: true }).data;

    return NextResponse.json({
      riskScores: parsedRiskScores,
      trends: parsedTrends,
    });
  } catch (error) {
    console.error('Error reading crime data:', error);
    return NextResponse.json({ error: 'Failed to load crime data' }, { status: 500 });
  }
}
